### Convert scanned recipes to json recipe schema

## Setup

Clone
```bash
git clone https://github.com/Elektrobarista/recipe-ocr-ai.git
```

1. **Install dependencies**
   ```bash
   cd recipe-ocr-ai
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
  python -m cli.main \
    --image /path/to/recipe.jpg \
    --output-dir ./output
  ```

  - Loads the prompt from `CHATGPT_PROMPT` in your `.env`.
  - Sends the image + prompt to the ChatGPT API.
  - Saves a pure schema.org Recipe JSON as `<image-stem>-<timestamp>.json` in `./output`.

- **Entire Folder with Images**

```bash
python -m cli.main \
  --image-dir ./images \
  --output-dir ./output \
  --concurrency 2
```

  - Finds all `*.jpg`, `*.jpeg`, `*.png` in the folder.
  - Creates a separate schema.org Recipe JSON for each file in the output folder.

By default, up to 2 images are sent in parallel to the OpenAI API (`OPENAI_CONCURRENCY=2`). You can adjust the parallelism to your OpenAI rate limit using `--concurrency` or the environment variable `OPENAI_CONCURRENCY`.

Optionally, you can adjust the model used or the maximum response length with `--model` and `--max-tokens`.

### Mode 2: schema.org Recipe JSON → Tandoor JSON (+ optional import)

The tool automatically detects available features based on environment variables:

- **Automatic Tandoor Import**: If `TANDOOR_BASE_URL` and `TANDOOR_API_TOKEN` are set, recipes are automatically imported to Tandoor.
- **Automatic Image Generation**: If `OPENAI_API_KEY` is set, preview images are automatically generated for each recipe.

- **Simple Usage (with auto-detection)**

```bash
# Set up environment variables
export TANDOOR_BASE_URL="https://tandoor.example.com"
export TANDOOR_API_TOKEN="your-api-token"
export OPENAI_API_KEY="your-openai-api-key"

# Convert and import - everything happens automatically!
python -m cli.main --schema-dir ./output
```

  - Reads all `*.json` in `./output`.
  - Converts each to Tandoor JSON format.
  - **Automatically generates images** (if `OPENAI_API_KEY` is set).
  - **Automatically imports to Tandoor** (if credentials are set).
  - Images are saved locally and uploaded to Tandoor.

- **Dry Run (JSON only, no API calls)**

```bash
python -m cli.main \
  --schema-dir ./output \
  --dry-run
```

  - Only creates Tandoor JSON files, no API calls.
  - Useful for testing or manual review.

- **Disable Image Generation**

```bash
python -m cli.main \
  --schema-dir ./output \
  --no-images
```

  - Skips image generation even if `OPENAI_API_KEY` is set.

- **Custom Image Directory**

```bash
python -m cli.main \
  --schema-dir ./output \
  --image-output-dir ./images
```

  - Saves generated images to a custom directory (default: same as schema JSON directory).

**Note:** 
- Images are generated using DALL-E 3 and automatically uploaded to Tandoor when importing.
- You can use `--tandoor-base-url` and `--tandoor-token` as CLI arguments instead of environment variables.
- The upload uses multipart/form-data or a separate image upload endpoint, depending on Tandoor's API capabilities.

### Mode 3: Import Already Converted Tandoor JSONs Directly

If you already have Tandoor-compatible JSON files (e.g., from a previous dry run), you can import them without re-converting:

- **Import Single Tandoor JSON**

```bash
python -m cli.main \
  --tandoor-json ./output/recipe-tandoor.json \
  --tandoor-base-url https://tandoor.example.com \
  --tandoor-token your-api-token
```

- **Import Entire Folder with Tandoor JSONs**

```bash
python -m cli.main \
  --tandoor-json-dir ./output/tandoor-jsons \
  --tandoor-base-url https://tandoor.example.com \
  --tandoor-token your-api-token
```

With `--tandoor-dry-run` in this mode, only the files that would be imported are listed without making an API call.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
