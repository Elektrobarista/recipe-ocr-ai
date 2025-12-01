from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _parse_ingredient(item: str) -> Dict[str, str]:
    """
    Parse a free-form ingredient string into {food, amount, unit}.

    Tries to handle patterns like:
    - "125 g Ciabattabrötchen"
    - "Ciabattabrötchen: 125 g"
    - "Ciabattabrötchen"
    """

    text = item.strip()

    # Pattern 1: amount unit food  (e.g. "125 g Ciabattabrötchen")
    leading_pattern = re.compile(r"^\s*([\d.,]+)\s+(\S+)\s+(.+)$")
    m = leading_pattern.match(text)
    if m:
        amount, unit, food = m.groups()
        return {
            "food": food.strip(),
            "amount": amount.strip(),
            "unit": unit.strip(),
        }

    # Pattern 2: "Name: 125 g" (HelloFresh-stil mit Doppelpunkt)
    if ":" in text:
        name_part, rest = text.split(":", 1)
        name_part = name_part.strip()
        rest = rest.strip()
        m2 = leading_pattern.match(rest)
        amount = ""
        unit = ""
        if m2:
            amount, unit, _ = m2.groups()
        elif rest:
            # Fallback: alles als Einheit interpretieren
            unit = rest
        return {
            "food": name_part,
            "amount": amount.strip(),
            "unit": unit.strip(),
        }

    # Pattern 3: food amount unit  (e.g. "Ciabattabrötchen 125 g")
    trailing_pattern = re.compile(r"^(.+?)\s+([\d.,]+)\s+(\S+)\s*$")
    m3 = trailing_pattern.match(text)
    if m3:
        food, amount, unit = m3.groups()
        return {
            "food": food.strip(),
            "amount": amount.strip(),
            "unit": unit.strip(),
        }

    # Fallback: nur Name bekannt
    return {
        "food": text,
        "amount": "",
        "unit": "",
    }


def map_schema_to_tandoor(schema_recipe: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a schema.org/Recipe JSON object to a minimal Tandoor-compatible recipe JSON.
    """

    # --- Basic Fields ---
    tandoor: Dict[str, Any] = {
        "name": schema_recipe.get("name", ""),
        "description": schema_recipe.get("description", ""),
        "instructions": "",
        "ingredients": [],
        "steps": [],
        "servings": schema_recipe.get("recipeYield", ""),
        "source_url": schema_recipe.get("mainEntityOfPage", "")
        or schema_recipe.get("url", ""),
        "nutrition": {},
    }

    # --- Ingredients Mapping ---
    ingredients: List[Dict[str, str]] = []
    for item in schema_recipe.get("recipeIngredient", []) or []:
        if not isinstance(item, str):
            continue
        ingredients.append(_parse_ingredient(item))
    tandoor["ingredients"] = ingredients

    # --- Instructions Mapping ---
    steps: List[Dict[str, str]] = []
    for step in schema_recipe.get("recipeInstructions", []) or []:
        if isinstance(step, dict):
            text = (step.get("text") or "").strip()
        else:
            text = str(step).strip()
        if text:
            steps.append({"instruction": text})
    tandoor["steps"] = steps
    tandoor["instructions"] = "\n\n".join(s["instruction"] for s in steps)

    # --- Nutrition Mapping ---
    nutrition = schema_recipe.get("nutrition") or {}
    if isinstance(nutrition, dict):
        tandoor["nutrition"] = {
            "servingSize": nutrition.get("servingSize", ""),
            "calories": nutrition.get("calories", ""),
            "fatContent": nutrition.get("fatContent", ""),
            "saturatedFatContent": nutrition.get("saturatedFatContent", ""),
            "carbohydrateContent": nutrition.get("carbohydrateContent", ""),
            "sugarContent": nutrition.get("sugarContent", ""),
            "proteinContent": nutrition.get("proteinContent", ""),
            "saltContent": nutrition.get("saltContent", ""),
        }

    return tandoor


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a schema.org Recipe JSON file to a Tandoor-compatible JSON.",
    )
    parser.add_argument(
        "--schema-json",
        required=True,
        help="Path to the schema.org Recipe JSON file.",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Path for the Tandoor JSON output (default: <schema-stem>-tandoor.json).",
    )
    args = parser.parse_args()

    in_path = Path(args.schema_json).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"Input schema JSON not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    tandoor_json = map_schema_to_tandoor(schema)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_path = in_path.with_name(f"{in_path.stem}-tandoor.json")

    out_path.write_text(
        json.dumps(tandoor_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Tandoor JSON written to {out_path}")


if __name__ == "__main__":
    main()


