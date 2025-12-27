#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from cli.schema_to_tandoor import map_schema_to_tandoor

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send an image and prompt to the ChatGPT API and store the JSON "
            "response, or convert an existing schema.org Recipe JSON to "
            "Tandoor-compatible JSON."
        ),
    )
    image_group = parser.add_argument_group(
        "image-to-schema",
        "Generate schema.org Recipe JSON from an image or a folder of images",
    )
    image_group.add_argument(
        "--image",
        help="Path to the JPG or PNG image to analyze.",
    )
    image_group.add_argument(
        "--output-dir",
        help="Directory where the JSON response should be written.",
    )
    image_group.add_argument(
        "--image-dir",
        help="Directory containing JPG/PNG files to convert to schema.org Recipe JSON.",
    )

    tandoor_group = parser.add_argument_group(
        "schema-to-tandoor",
        "Convert existing schema.org Recipe JSON to Tandoor JSON and optionally import into Tandoor via API",
    )
    tandoor_group.add_argument(
        "--schema-json",
        help="Path to an existing schema.org Recipe JSON file to convert.",
    )
    tandoor_group.add_argument(
        "--tandoor-out",
        help="Output path for a single Tandoor JSON (default: <schema-stem>-tandoor.json).",
    )
    tandoor_group.add_argument(
        "--schema-dir",
        help="Directory containing schema.org Recipe JSON files to convert and import.",
    )
    tandoor_group.add_argument(
        "--tandoor-json",
        help="Path to an existing Tandoor JSON file to import directly.",
    )
    tandoor_group.add_argument(
        "--tandoor-json-dir",
        help="Directory containing Tandoor JSON files to import directly.",
    )
    parser.add_argument(
        "--tandoor-base-url",
        help="Tandoor base URL, e.g. https://tandoor.example.com (defaults to TANDOOR_BASE_URL env).",
    )
    parser.add_argument(
        "--tandoor-token",
        help="Tandoor API token (defaults to TANDOOR_API_TOKEN env).",
    )
    parser.add_argument(
        "--tandoor-dry-run",
        action="store_true",
        help="Only write Tandoor JSON files, do not POST to the Tandoor API.",
    )
    parser.add_argument(
        "--api",
        choices=["openai", "gemini"],
        default=None,
        help="API provider to use (defaults to API_PROVIDER env or openai).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name (defaults to OPENAI_MODEL/GEMINI_MODEL env or provider default).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum completion tokens in the response (optional).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=(
            "Maximum number of parallel API requests when processing a folder of "
            "images (default from OPENAI_CONCURRENCY/GEMINI_CONCURRENCY env or 2)."
        ),
    )
    return parser.parse_args()


def validate_image_path(image_path: Path) -> None:
    if not image_path.exists():
        raise SystemExit(f"Image file not found: {image_path}")
    if not image_path.is_file():
        raise SystemExit(f"Image path is not a file: {image_path}")
    if image_path.suffix.lower() not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise SystemExit(f"Unsupported image type '{image_path.suffix}'. Allowed: {allowed}")


def load_config(provider: str) -> tuple[str, str]:
    load_dotenv()
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        prompt = os.getenv("GEMINI_PROMPT") or os.getenv("CHATGPT_PROMPT")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY missing. Add it to your environment or .env file.")
    else:  # openai
        api_key = os.getenv("OPENAI_API_KEY")
        prompt = os.getenv("CHATGPT_PROMPT")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY missing. Add it to your environment or .env file.")
    if not prompt:
        raise SystemExit("CHATGPT_PROMPT or GEMINI_PROMPT missing. Add it to your environment or .env file.")
    return api_key, prompt


def build_image_payload(image_path: Path) -> dict:
    data = image_path.read_bytes()
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
    }


def build_gemini_image_payload(image_path: Path) -> dict:
    data = image_path.read_bytes()
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "inline_data": {
            "mime_type": mime,
            "data": encoded,
        }
    }


def extract_recipe_json(response: dict) -> dict:
    choices = response.get("choices")
    if not choices:
        raise SystemExit("OpenAI response did not contain any choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_json" and isinstance(part.get("json"), dict):
                return part["json"]
            if part.get("type") == "text" and part.get("text"):
                try:
                    return json.loads(part["text"])
                except json.JSONDecodeError:
                    continue
    elif isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise SystemExit("Model returned non-JSON text.") from exc

    raise SystemExit("OpenAI response did not contain parsable JSON content.")


def call_openai(
    api_key: str,
    prompt: str,
    image_content: dict,
    model: str,
    max_tokens: int | None,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content,
                ],
            }
        ],
    }
    if max_tokens is not None:
        payload["max_completion_tokens"] = max_tokens
    payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        OPENAI_API_URL,
        headers=headers,
        json=payload,
        timeout=300,
    )
    if response.status_code != 200:
        raise SystemExit(
            f"OpenAI API error {response.status_code}: {response.text}",
        )
    return response.json()


def call_gemini(
    api_key: str,
    prompt: str,
    image_content: dict,
    model: str,
    max_tokens: int | None,
) -> dict:
    url = f"{GEMINI_API_BASE_URL}/models/{model}:generateContent"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    image_content,
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
        },
    }
    if max_tokens is not None:
        payload["generationConfig"]["maxOutputTokens"] = max_tokens
    headers = {
        "Content-Type": "application/json",
    }
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        params={"key": api_key},
        timeout=300,
    )
    if response.status_code != 200:
        raise SystemExit(
            f"Gemini API error {response.status_code}: {response.text}",
        )
    return response.json()


def extract_recipe_json_gemini(response: dict) -> dict:
    candidates = response.get("candidates")
    if not candidates:
        raise SystemExit("Gemini response did not contain any candidates.")
    content = candidates[0].get("content") or {}
    parts = content.get("parts", [])
    
    for part in parts:
        text = part.get("text")
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit("Model returned non-JSON text.") from exc
    
    raise SystemExit("Gemini response did not contain parsable JSON content.")


def build_output_path(image_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Nutze Mikrosekunden im Timestamp, um Kollisionen bei paralleler Verarbeitung zu vermeiden
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    base_name = f"{image_path.stem}-{timestamp}.json"
    return output_dir / base_name


def _process_image(
    image_path: Path,
    output_dir: Path,
    api_key: str,
    prompt: str,
    model: str,
    max_tokens: int | None,
    provider: str,
    progress_task: int | None = None,
    progress: Progress | None = None,
) -> None:
    if progress and progress_task is not None:
        progress.update(
            progress_task,
            description=f"Verarbeite Rezept - {image_path.name}",
        )
    validate_image_path(image_path)

    if provider == "gemini":
        image_payload = build_gemini_image_payload(image_path)
        api_name = "Gemini"
    else:  # openai
        image_payload = build_image_payload(image_path)
        api_name = "ChatGPT"

    if progress and progress_task is not None:
        progress.update(progress_task, description=f"{api_name} analysiert {image_path.name}...")

    if provider == "gemini":
        result = call_gemini(
            api_key=api_key,
            prompt=prompt,
            image_content=image_payload,
            model=model,
            max_tokens=max_tokens,
        )
        recipe_json = extract_recipe_json_gemini(result)
    else:  # openai
        result = call_openai(
            api_key=api_key,
            prompt=prompt,
            image_content=image_payload,
            model=model,
            max_tokens=max_tokens,
        )
        recipe_json = extract_recipe_json(result)

    output_path = build_output_path(image_path, output_dir)
    output_path.write_text(
        json.dumps(recipe_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if progress and progress_task is not None:
        progress.update(progress_task, description=f"✓ {image_path.name} abgeschlossen")
    else:
        print(f"Response written to {output_path}")


def _iter_image_files(image_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not image_dir.exists():
        raise SystemExit(f"Image directory or file not found: {image_dir}")
    
    # Wenn es eine Textdatei ist, zeilenweise Bildpfade einlesen
    if image_dir.is_file():
        try:
            with image_dir.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    img_path = Path(line).expanduser().resolve()
                    if img_path.exists() and img_path.is_file():
                        if img_path.suffix.lower() in ALLOWED_SUFFIXES:
                            files.append(img_path)
                        else:
                            print(
                                f"Warnung: Überspringe {img_path} (nicht unterstütztes Format)",
                                file=sys.stderr,
                            )
                    else:
                        print(
                            f"Warnung: Bild nicht gefunden: {img_path}",
                            file=sys.stderr,
                        )
        except Exception as exc:
            raise SystemExit(f"Fehler beim Lesen der Bildliste {image_dir}: {exc}") from exc
        if not files:
            raise SystemExit(f"Keine gültigen Bildpfade in {image_dir} gefunden")
        return files
    
    # Wenn es ein Verzeichnis ist, wie bisher alle Bilder finden
    if not image_dir.is_dir():
        raise SystemExit(f"Image path is neither a directory nor a text file: {image_dir}")
    for suffix in ALLOWED_SUFFIXES:
        files.extend(sorted(image_dir.glob(f"*{suffix}")))
    return files


def _write_error_log(
    failed_images: list[tuple[Path, Exception]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Schreibe detailliertes Fehler-Log und failed-images.txt.

    Returns:
        Tuple von (error_log_path, failed_images_path)
    """
    # Stelle sicher, dass das Output-Verzeichnis existiert
    # (kann fehlen, wenn Fehler vor build_output_path auftreten)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Nutze Mikrosekunden im Timestamp, um Kollisionen bei paralleler Verarbeitung zu vermeiden
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    error_log_path = output_dir / f"errors-{timestamp}.log"
    failed_images_path = output_dir / f"failed-images-{timestamp}.txt"

    # Detailliertes Fehler-Log schreiben
    with error_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"=== Error Log: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} ===\n\n")
        for img_path, exc in failed_images:
            log_file.write(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {img_path}\n")
            log_file.write(f"  Error: {type(exc).__name__}: {exc}\n")
            log_file.write("  Traceback:\n")
            # Stacktrace formatieren
            tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
            for line in tb_lines:
                log_file.write(f"    {line}")
            log_file.write("\n")

    # Einfache Liste der fehlgeschlagenen Bildpfade
    with failed_images_path.open("w", encoding="utf-8") as failed_file:
        for img_path, _ in failed_images:
            failed_file.write(f"{img_path}\n")

    return error_log_path, failed_images_path


def _iter_schema_files(schema_dir: Path) -> list[Path]:
    path = schema_dir
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"Schema JSON directory not found: {path}")
    return sorted(path.glob("*.json"))


def _load_tandoor_config(args: argparse.Namespace) -> tuple[str, str]:
    load_dotenv()
    base_url = args.tandoor_base_url or os.getenv("TANDOOR_BASE_URL") or ""
    token = args.tandoor_token or os.getenv("TANDOOR_API_TOKEN") or ""
    if not base_url:
        raise SystemExit("TANDOOR_BASE_URL is required (env or --tandoor-base-url).")
    if not token:
        raise SystemExit("TANDOOR_API_TOKEN is required (env or --tandoor-token).")
    return base_url.rstrip("/"), token


def _send_to_tandoor(recipe: dict, base_url: str, token: str) -> bool:
    url = f"{base_url}/api/recipe/"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    response = requests.post(
        url,
        headers=headers,
        json=recipe,
        timeout=(10, 300),
    )
    if response.status_code in (200, 201):
        return True

    print(
        f"Failed to import recipe '{recipe.get('name', '')}' "
        f"({response.status_code}): {response.text}",
        file=sys.stderr,
    )
    return False


def main() -> None:
    args = parse_args()
    has_image_mode = bool(args.image or args.image_dir)
    has_schema_mode = bool(args.schema_json or args.schema_dir)
    has_tandoor_import_mode = bool(args.tandoor_json or args.tandoor_json_dir)

    active_modes = sum(
        1 for flag in (has_image_mode, has_schema_mode, has_tandoor_import_mode) if flag
    )
    if active_modes != 1:
        raise SystemExit(
            "Choose exactly one mode:\n"
            "- image-to-schema (--image/--image-dir)\n"
            "- schema-to-tandoor (--schema-json/--schema-dir)\n"
            "- tandoor-import (--tandoor-json/--tandoor-json-dir)",
        )

    # Mode 1: schema.org Recipe JSON -> Tandoor JSON (+ optional import)
    if has_schema_mode:
        if args.schema_json and args.schema_dir:
            raise SystemExit("Use either --schema-json or --schema-dir, not both.")

        if args.schema_dir:
            schema_dir = Path(args.schema_dir).expanduser().resolve()
            schema_files = _iter_schema_files(schema_dir)
            if not schema_files:
                raise SystemExit(
                    f"No schema.org Recipe JSON files found in directory: {schema_dir}",
                )
        else:
            schema_path = Path(args.schema_json).expanduser().resolve()
            if not schema_path.exists():
                raise SystemExit(f"Schema JSON file not found: {schema_path}")
            schema_files = [schema_path]

        base_url = token = ""
        if not args.tandoor_dry_run:
            base_url, token = _load_tandoor_config(args)

        for schema_file in schema_files:
            with schema_file.open("r", encoding="utf-8") as f:
                schema_recipe = json.load(f)
            tandoor = map_schema_to_tandoor(schema_recipe)

            if args.tandoor_out and len(schema_files) == 1:
                out_path = Path(args.tandoor_out).expanduser().resolve()
            else:
                out_path = schema_file.with_name(f"{schema_file.stem}-tandoor.json")

            out_path.write_text(
                json.dumps(tandoor, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Tandoor JSON written to {out_path}")

            if not args.tandoor_dry_run:
                _send_to_tandoor(tandoor, base_url=base_url, token=token)

        return

    # Mode 2: bereits konvertierte Tandoor JSON direkt importieren
    if has_tandoor_import_mode:
        if args.tandoor_json and args.tandoor_json_dir:
            raise SystemExit("Use either --tandoor-json or --tandoor-json-dir, not both.")

        if args.tandoor_dry_run:
            # Kein API-Call, nur anzeigen, was importiert würde.
            if args.tandoor_json_dir:
                base_dir = Path(args.tandoor_json_dir).expanduser().resolve()
                files = sorted(base_dir.glob("*.json"))
            else:
                files = [Path(args.tandoor_json).expanduser().resolve()]

            for tandoor_file in files:
                print(f"[dry-run] Would import Tandoor JSON: {tandoor_file}")
            return

        base_url, token = _load_tandoor_config(args)

        if args.tandoor_json_dir:
            base_dir = Path(args.tandoor_json_dir).expanduser().resolve()
            files = sorted(base_dir.glob("*.json"))
            if not files:
                raise SystemExit(f"No Tandoor JSON files found in directory: {base_dir}")
        else:
            tandoor_path = Path(args.tandoor_json).expanduser().resolve()
            if not tandoor_path.exists():
                raise SystemExit(f"Tandoor JSON file not found: {tandoor_path}")
            files = [tandoor_path]

        for tandoor_file in files:
            with tandoor_file.open("r", encoding="utf-8") as f:
                recipe = json.load(f)
            ok = _send_to_tandoor(recipe, base_url=base_url, token=token)
            if ok:
                print(f"Imported Tandoor JSON from {tandoor_file}")

        return

    # Mode 3: image -> schema.org Recipe JSON via ChatGPT
    if not has_image_mode or not args.output_dir:
        raise SystemExit(
            "For image processing, provide --output-dir and either --image or --image-dir.",
        )

    if args.image and args.image_dir:
        raise SystemExit("Use either --image or --image-dir, not both.")

    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.image_dir:
        image_dir = Path(args.image_dir).expanduser().resolve()
        image_files = _iter_image_files(image_dir)
        if not image_files:
            raise SystemExit(f"No JPG/PNG files found in directory: {image_dir}")
    else:
        image_path = Path(args.image).expanduser().resolve()
        image_files = [image_path]

    # Determine API provider
    provider = args.api or os.getenv("API_PROVIDER", "openai")
    if provider not in ("openai", "gemini"):
        raise SystemExit(f"Invalid API provider: {provider}. Must be 'openai' or 'gemini'.")

    api_key, prompt = load_config(provider)

    # Set default model based on provider
    if args.model:
        model = args.model
    elif provider == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    else:  # openai
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Begrenze parallele Requests, um Rate Limits berücksichtigen zu können.
    # Dabei explizit zwischen None (nicht gesetzt) und 0 unterscheiden.
    if args.concurrency is not None:
        concurrency = args.concurrency
    else:
        env_key = "GEMINI_CONCURRENCY" if provider == "gemini" else "OPENAI_CONCURRENCY"
        concurrency = int(os.getenv(env_key, "2"))
    concurrency = max(1, concurrency)

    console = Console()
    num_images = len(image_files)

    # Progress-Anzeige mit Rich
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task(
            f"Verarbeite {num_images} Rezeptbilder...",
            total=num_images,
        )

        if len(image_files) == 1 or concurrency == 1:
            # Einzelne Verarbeitung
            for idx, image_path in enumerate(image_files):
                task_id = progress.add_task(
                    f"Verarbeite Rezept {idx + 1}/{num_images} - {image_path.name}",
                    total=1,
                )
                try:
                    _process_image(
                        image_path=image_path,
                        output_dir=output_dir,
                        api_key=api_key,
                        prompt=prompt,
                        model=model,
                        max_tokens=args.max_tokens,
                        provider=provider,
                        progress_task=task_id,
                        progress=progress,
                    )
                    progress.update(task_id, completed=1)
                    progress.update(main_task, advance=1)
                except Exception as exc:  # noqa: BLE001
                    progress.update(
                        task_id,
                        description=f"✗ Fehler: {image_path.name}",
                        completed=1,
                    )
                    progress.update(main_task, advance=1)
                    # Fehler-Log schreiben
                    error_log_path, failed_images_path = _write_error_log(
                        [(image_path, exc)],
                        output_dir,
                    )
                    console.print(f"\n[red]Fehler beim Verarbeiten von {image_path.name}:[/red] {exc}")
                    console.print(f"[yellow]Fehlgeschlagenes Bild:[/yellow] {failed_images_path}")
                    console.print(f"[yellow]Detaillierte Fehler:[/yellow] {error_log_path}")
                    raise SystemExit(1) from exc
        else:
            # Parallele Verarbeitung
            had_errors = False
            failed_images: list[tuple[Path, Exception]] = []
            task_map: dict[Path, int] = {}

            # Tasks für alle Bilder erstellen
            for idx, image_path in enumerate(image_files):
                task_id = progress.add_task(
                    f"Warte auf Verarbeitung - {image_path.name}",
                    total=1,
                )
                task_map[image_path] = task_id

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_image = {
                    executor.submit(
                        _process_image,
                        image_path=image_path,
                        output_dir=output_dir,
                        api_key=api_key,
                        prompt=prompt,
                        model=model,
                        max_tokens=args.max_tokens,
                        provider=provider,
                        progress_task=task_map[image_path],
                        progress=progress,
                    ): image_path
                    for image_path in image_files
                }

                for future in as_completed(future_to_image):
                    img = future_to_image[future]
                    task_id = task_map[img]
                    try:
                        future.result()
                        progress.update(task_id, completed=1)
                        progress.update(main_task, advance=1)
                    except Exception as exc:  # noqa: BLE001
                        had_errors = True
                        failed_images.append((img, exc))
                        progress.update(
                            task_id,
                            description=f"✗ Fehler: {img.name}",
                            completed=1,
                        )
                        progress.update(main_task, advance=1)

            if had_errors:
                # Fehler-Logs schreiben
                error_log_path, failed_images_path = _write_error_log(
                    failed_images,
                    output_dir,
                )
                console.print("\n[red]Fehler beim Verarbeiten einiger Bilder:[/red]")
                for img, exc in failed_images:
                    console.print(f"  [red]✗[/red] {img}: {exc}")
                console.print(f"\n[yellow]Fehlgeschlagene Bilder:[/yellow] {failed_images_path}")
                console.print(f"[yellow]Detaillierte Fehler:[/yellow] {error_log_path}")
                console.print(
                    f"\n[yellow]Tipp:[/yellow] Erneut versuchen mit: "
                    f'python -m cli.main --image-dir "{failed_images_path}" --output-dir "{output_dir}"'
                )
                raise SystemExit(1)

    # Erfolgsmeldung
    console.print(f"\n[green]✓ Fertig! {num_images} Rezept(e) erfolgreich verarbeitet.[/green]")


if __name__ == "__main__":
    main()

