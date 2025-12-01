#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from .schema_to_tandoor import map_schema_to_tandoor

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


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
        "--model",
        default=None,
        help="Override the model name (defaults to OPENAI_MODEL env or gpt-4o-mini).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum completion tokens in the response (optional).",
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


def load_config() -> tuple[str, str]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    prompt = os.getenv("CHATGPT_PROMPT")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY missing. Add it to your environment or .env file.")
    if not prompt:
        raise SystemExit("CHATGPT_PROMPT missing. Add it to your environment or .env file.")
    return api_key, prompt


def build_image_payload(image_path: Path) -> dict:
    data = image_path.read_bytes()
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
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


def build_output_path(image_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base_name = f"{image_path.stem}-{timestamp}.json"
    return output_dir / base_name


def _iter_image_files(image_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not image_dir.exists() or not image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {image_dir}")
    for suffix in ALLOWED_SUFFIXES:
        files.extend(sorted(image_dir.glob(f"*{suffix}")))
    return files


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


def _send_to_tandoor(recipe: dict, base_url: str, token: str) -> None:
    url = f"{base_url}/api/recipe/"
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    response = requests.post(url, headers=headers, json=recipe, timeout=60)
    if response.status_code not in (200, 201):
        print(
            f"Failed to import recipe '{recipe.get('name', '')}' "
            f"({response.status_code}): {response.text}",
            file=sys.stderr,
        )


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
            _send_to_tandoor(recipe, base_url=base_url, token=token)
            print(f"Imported Tandoor JSON from {tandoor_file}")

        return

    # Mode 3: image -> schema.org Recipe JSON via ChatGPT
    if not has_image_mode or not args.output_dir:
        raise SystemExit(
            "For image processing, provide --output-dir and either --image or --image-dir.",
        )

    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.image_dir:
        image_dir = Path(args.image_dir).expanduser().resolve()
        image_files = _iter_image_files(image_dir)
        if not image_files:
            raise SystemExit(f"No JPG/PNG files found in directory: {image_dir}")
    else:
        image_path = Path(args.image).expanduser().resolve()
        image_files = [image_path]

    api_key, prompt = load_config()
    model = args.model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    for image_path in image_files:
        validate_image_path(image_path)
        image_payload = build_image_payload(image_path)

        result = call_openai(
            api_key=api_key,
            prompt=prompt,
            image_content=image_payload,
            model=model,
            max_tokens=args.max_tokens,
        )

        output_path = build_output_path(image_path, output_dir)
        recipe_json = extract_recipe_json(result)
        output_path.write_text(
            json.dumps(recipe_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"Response written to {output_path}")


if __name__ == "__main__":
    main()

