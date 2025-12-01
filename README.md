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

There are three distinct modes that are mutually exclusive:

### Mode 1: Image(s) → schema.org Recipe JSON

- **Single Image**

  ```bash
  python cli/main.py \
    --image /path/to/recipe.jpg \
    --output-dir ./output
  ```

  - Loads the prompt from `CHATGPT_PROMPT` in your `.env`.
  - Sends the image + prompt to the ChatGPT API.
  - Saves a pure schema.org Recipe JSON as `<image-stem>-<timestamp>.json` in `./output`.

- **Entire Folder with Images**

  ```bash
  python cli/main.py \
    --image-dir ./images \
    --output-dir ./output \
    --concurrency 2
  ```

  - Finds all `*.jpg`, `*.jpeg`, `*.png` in the folder.
  - Creates a separate schema.org Recipe JSON for each file in the output folder.

By default, up to 2 images are sent in parallel to the OpenAI API (`OPENAI_CONCURRENCY=2`). You can adjust the parallelism to your OpenAI rate limit using `--concurrency` or the environment variable `OPENAI_CONCURRENCY`.

Optionally, you can adjust the model used or the maximum response length with `--model` and `--max-tokens`.

### Mode 2: schema.org Recipe JSON → Tandoor JSON (+ optional import)

- **Convert Single File (without import)**

  ```bash
  python cli/main.py \
    --schema-json ./output/recipe.json \
    --tandoor-dry-run
  ```

  - Reads a schema.org Recipe JSON.
  - Converts it to a Tandoor-compatible JSON.
  - Writes `<recipe-stem>-tandoor.json` next to the input file.

- **Convert Entire Folder and Import to Tandoor**

  ```bash
  export TANDOOR_BASE_URL="https://tandoor.example.com"
  export TANDOOR_API_TOKEN="your-api-token"

  python cli/main.py \
    --schema-dir ./output
  ```

  - Reads all `*.json` in `./output`.
  - Creates `<name>-tandoor.json` for each.
  - POSTs each recipe to `"$TANDOOR_BASE_URL/api/recipe/"`.

Alternatively, `--tandoor-base-url` and `--tandoor-token` can be used as CLI arguments instead of environment variables. With `--tandoor-dry-run`, only the Tandoor JSON files are created without making an API call.

### Mode 3: Import Already Converted Tandoor JSONs Directly

If you already have Tandoor-compatible JSON files (e.g., from a previous dry run), you can import them without re-converting:

- **Import Single Tandoor JSON**

  ```bash
  python cli/main.py \
    --tandoor-json ./output/recipe-tandoor.json \
    --tandoor-base-url https://tandoor.example.com \
    --tandoor-token your-api-token
  ```

- **Import Entire Folder with Tandoor JSONs**

  ```bash
  python cli/main.py \
    --tandoor-json-dir ./output/tandoor-jsons \
    --tandoor-base-url https://tandoor.example.com \
    --tandoor-token your-api-token
  ```

With `--tandoor-dry-run` in this mode, only the files that would be imported are listed without making an API call.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
