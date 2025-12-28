from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_number(value: str | float | int) -> Optional[float]:
    """Extract the first numeric value from a string and convert to float.
    
    Also handles numeric types (int, float) directly.
    """
    # If value is already a number, return it as float
    if isinstance(value, (int, float)):
        return float(value)
    
    # If value is not a string, try to convert it
    if not isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    # If value is empty string, return None
    if not value.strip():
        return None
    
    # Unterstützt auch Formate mit führendem Dezimaltrennzeichen wie ".5"
    match = re.search(r"[-+]?(?:\d+[.,]?\d*|\d*[.,]\d+)", value)
    if not match:
        return None
    num = match.group(0).replace(",", ".")
    try:
        return float(num)
    except ValueError:
        return None


def _parse_ingredient(item: str) -> Dict[str, Any]:
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
            "amount": _parse_number(amount) or 0.0,
            "unit": unit.strip(),
        }

    # Pattern 2: "Name: 125 g" (HelloFresh-stil mit Doppelpunkt)
    if ":" in text:
        name_part, rest = text.split(":", 1)
        name_part = name_part.strip()
        rest = rest.strip()
        amount = ""
        unit = ""

        # Spezieller Parser für "<amount> <unit>" (z. B. "125 g" oder "2 EL Olivenöl")
        parts = rest.split()
        if parts:
            amount_candidate = parts[0]
            if _parse_number(amount_candidate) is not None and len(parts) >= 2:
                amount = amount_candidate
                unit = " ".join(parts[1:])
            else:
                # Fallback: komplette Rest-Zeile als Einheit behandeln
                unit = rest

        return {
            "food": name_part,
            "amount": _parse_number(amount) or 0.0,
            "unit": unit.strip(),
        }

    # Pattern 3: food amount unit  (e.g. "Ciabattabrötchen 125 g")
    trailing_pattern = re.compile(r"^(.+?)\s+([\d.,]+)\s+(\S+)\s*$")
    m3 = trailing_pattern.match(text)
    if m3:
        food, amount, unit = m3.groups()
        return {
            "food": food.strip(),
            "amount": _parse_number(amount) or 0.0,
            "unit": unit.strip(),
        }

    # Fallback: nur Name bekannt, Menge = 1, Einheit leer
    return {
        "food": text,
        "amount": 1.0,
        "unit": "",
    }


def map_schema_to_tandoor(schema_recipe: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a schema.org/Recipe JSON object to a minimal Tandoor-compatible recipe JSON.
    """

    # --- Basic Fields ---
    tandoor: Dict[str, Any] = {
        "name": (schema_recipe.get("name") or "").strip(),
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
    ingredients: List[Dict[str, Any]] = []
    for item in schema_recipe.get("recipeIngredient", []) or []:
        if not isinstance(item, str):
            continue
        ingredients.append(_parse_ingredient(item))
    tandoor["ingredients"] = ingredients

    # --- Instructions Mapping ---
    steps: List[Dict[str, Any]] = []
    raw_steps = schema_recipe.get("recipeInstructions", []) or []

    # recipeInstructions kann entweder eine Liste von Schritten ODER ein einzelner String sein.
    # Wenn es ein String oder ein einzelnes Objekt ist, in eine Liste verpacken,
    # damit wir nicht versehentlich Zeichen/Keys iterieren.
    if isinstance(raw_steps, (str, dict)):
        raw_steps = [raw_steps]

    # Baue eine gemeinsame Zutatenliste, die bei Bedarf einem Step zugeordnet wird.
    common_step_ingredients: List[Dict[str, Any]] = []
    for ing in ingredients:
        food_name = ing["food"]
        unit_name = ing["unit"] or "Stück"
        amount = ing["amount"] or 0.0
        common_step_ingredients.append(
            {
                "food": {
                    "name": food_name,
                },
                "unit": {
                    "name": unit_name,
                },
                "amount": amount,
            }
        )

    first_step_assigned = False
    for step in raw_steps:
        if isinstance(step, dict):
            text = (step.get("text") or "").strip()
            name = (step.get("name") or "").strip()
        else:
            text = str(step).strip()
            name = ""
        if not text:
            continue

        # Nur dem ersten NICHT-LEEREN Schritt die vollständige Zutatenliste geben,
        # damit sie nicht in jedem Schritt dupliziert wird.
        if not first_step_assigned:
            step_ingredients = common_step_ingredients
            first_step_assigned = True
        else:
            step_ingredients = []

        steps.append(
            {
                "name": name,
                "instruction": text,
                "ingredients": step_ingredients,
            }
        )

    tandoor["steps"] = steps
    tandoor["instructions"] = "\n\n".join(s["instruction"] for s in steps)

    # --- Nutrition Mapping (Tandoor expects numeric macros) ---
    nutrition = schema_recipe.get("nutrition") or {}
    carbs_src = None
    fats_src = None
    proteins_src = None
    cals_src = None
    if isinstance(nutrition, dict):
        carbs_src = nutrition.get("carbohydrateContent")
        fats_src = nutrition.get("fatContent")
        proteins_src = nutrition.get("proteinContent")
        cals_src = nutrition.get("calories")

    tandoor["nutrition"] = {
        # Tandoor expects numeric macros; fallback auf 0.0 wenn unbekannt
        "carbohydrates": _parse_number(carbs_src) or 0.0,
        "fats": _parse_number(fats_src) or 0.0,
        "proteins": _parse_number(proteins_src) or 0.0,
        "calories": _parse_number(cals_src) or 0.0,
    }

    # --- Servings (recipeYield -> int) ---
    yield_str = str(schema_recipe.get("recipeYield", "") or "")
    servings_num = _parse_number(yield_str) or 0.0
    tandoor["servings"] = int(servings_num) if servings_num > 0 else 0

    # Sicherstellen, dass ein Name aus dem Inhalt stammt
    if not tandoor["name"]:
        raise SystemExit(
            "schema.org Recipe JSON enthält keinen 'name'. "
            "Passe den Prompt so an, dass immer ein Rezeptname ausgegeben wird, "
            "und generiere die JSON-Datei neu.",
        )

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


