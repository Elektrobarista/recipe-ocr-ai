#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send an image and prompt to the ChatGPT API and store the JSON response.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the JPG or PNG image to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the JSON response should be written.",
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


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    validate_image_path(image_path)
    api_key, prompt = load_config()
    model = args.model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
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

