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

Run the CLI with the image to analyze and a target folder for responses:

```bash
python cli/main.py \
  --image /path/to/recipe.jpg \
  --output-dir ./output
```

The script loads the prompt from `CHATGPT_PROMPT` in your `.env`, sends the image to the ChatGPT API, and stores the pure Schema.org recipe JSON (named `<image-stem>-<timestamp>.json`) in the specified output directory. Use `--model` or `--max-tokens` to override defaults when needed.