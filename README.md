### Convert scanned recipes to json recipe schema

## Setup

1. **Install dependencies**
   ```bash
   cd recipe_ai
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure environment**
   ```bash
   cp .env.example .env
   # edit .env with your OPENAI_API_KEY and prompt text
   ```

## Usage

Es gibt zwei klar getrennte Modi, die sich gegenseitig ausschließen:

### Modus 1: Bild(er) → schema.org Recipe JSON

- **Einzelnes Bild**

  ```bash
  python cli/main.py \
    --image /path/to/recipe.jpg \
    --output-dir ./output
  ```

  - Lädt den Prompt aus `CHATGPT_PROMPT` in deiner `.env`.
  - Schickt das Bild + Prompt an die ChatGPT API.
  - Speichert ein reines schema.org Recipe JSON als `<image-stem>-<timestamp>.json` in `./output`.

- **Ganze Ordner mit Bildern**

  ```bash
  python cli/main.py \
    --image-dir ./images \
    --output-dir ./output \
    --concurrency 2
  ```

  - Findet alle `*.jpg`, `*.jpeg`, `*.png` im Ordner.
  - Erzeugt für jede Datei ein eigenes schema.org Recipe JSON im Output-Ordner.

Standardmäßig werden bis zu 2 Bilder parallel gegen die OpenAI API geschickt (`OPENAI_CONCURRENCY=2`). Mit `--concurrency` oder der Umgebungsvariable `OPENAI_CONCURRENCY` kannst du die Parallelität an dein OpenAI-Rate-Limit anpassen.

Optional können mit `--model` und `--max-tokens` das verwendete Modell bzw. die maximale Antwortlänge angepasst werden.

### Modus 2: schema.org Recipe JSON → Tandoor JSON (+ optionaler Import)

- **Einzelne Datei konvertieren (ohne Import)**

  ```bash
  python cli/main.py \
    --schema-json ./output/recipe.json \
    --tandoor-dry-run
  ```

  - Liest ein schema.org Recipe JSON.
  - Wandelt es in ein Tandoor-kompatibles JSON um.
  - Schreibt `<recipe-stem>-tandoor.json` neben die Eingabedatei.

- **Ganzen Ordner konvertieren und nach Tandoor importieren**

  ```bash
  export TANDOOR_BASE_URL="https://tandoor.example.com"
  export TANDOOR_API_TOKEN="dein-api-token"

  python cli/main.py \
    --schema-dir ./output
  ```

  - Liest alle `*.json` in `./output`.
  - Erzeugt jeweils `<name>-tandoor.json`.
  - POSTet jedes Rezept an `"$TANDOOR_BASE_URL/api/recipe/"`.

Alternativ können `--tandoor-base-url` und `--tandoor-token` als CLI-Argumente statt Umgebungsvariablen genutzt werden. Mit `--tandoor-dry-run` werden nur die Tandoor-JSON-Dateien erzeugt, ohne API-Aufruf.

### Modus 3: Bereits konvertierte Tandoor JSONs direkt importieren

Wenn du schon Tandoor-kompatible JSON-Dateien (z. B. durch einen vorherigen Dry-Run) hast, kannst du sie ohne erneute Konvertierung importieren:

- **Einzelne Tandoor-JSON importieren**

  ```bash
  python cli/main.py \
    --tandoor-json ./output/recipe-tandoor.json \
    --tandoor-base-url https://tandoor.example.com \
    --tandoor-token dein-api-token
  ```

- **Ganzen Ordner mit Tandoor-JSONs importieren**

  ```bash
  python cli/main.py \
    --tandoor-json-dir ./output/tandoor-jsons \
    --tandoor-base-url https://tandoor.example.com \
    --tandoor-token dein-api-token
  ```

Mit `--tandoor-dry-run` werden in diesem Modus nur die Dateien aufgelistet, die importiert würden, ohne einen API-Aufruf zu machen.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.