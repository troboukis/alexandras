# Alexandras Data Pipeline

This repository collects, summarizes, and publishes Greek government decisions from the Diavgeia API related to the Alexandras refugee apartment buildings (Προσφυγικά Λεωφόρου Αλεξάνδρας).

It includes:

- `get_data.ipynb`: the original exploratory notebook / prototype pipeline
- `update_alexandras_data.py`: the production-style incremental updater script
- `publish_alexandras_update.sh`: a convenience script to run the update and publish to the GitHub Pages repo
- `index.html`: frontend page that reads and displays the dataset
- `data_alexandras.csv`: main dataset used by the page

## What the project does

The pipeline:

1. Queries the Diavgeia API for decisions likely related to the Alexandras refugee buildings.
2. Downloads document PDFs (cached locally under `data/pdfs/`).
3. Extracts text from PDFs with PyMuPDF.
4. Sends the text to the OpenAI API for structured summarization.
5. Cleans and normalizes the summary JSON.
6. Appends only new relevant rows to `data_alexandras.csv` (by default).

## Important behavior (new data only)

The updater is designed to avoid reprocessing old data by default.

- Existing rows in `data_alexandras.csv` are loaded only for comparison + final merge.
- Existing rows are **not** re-analyzed and **not** re-cleaned unless you explicitly pass `--clean-existing`.
- The script inspects newly fetched Diavgeia results and only processes rows that are new at the metadata-row level.
- Multiple rows with the same `ada` are allowed if other columns differ.
- Final save uses `drop_duplicates()` (full-row exact duplicates only), not dedupe by `ada`.

## Repository structure

- `update_alexandras_data.py`: incremental updater
- `publish_alexandras_update.sh`: update + copy + git add/commit/push helper
- `get_data.ipynb`: original notebook implementation
- `data/`
- `data/pdfs/`: cached PDF files
- `logs/`: updater logs (timestamped, unless disabled)
- `data_alexandras.csv`: final dataset consumed by `index.html`

## Requirements

- Python 3.11+ recommended
- OpenAI API key (`OPENAI_API_KEY`)
- Network access to:
  - `diavgeia.gov.gr`
  - OpenAI API

Python dependencies are listed in `requirements.txt` and include:

- `pandas`
- `requests`
- `openai`
- `PyMuPDF` (imported as `fitz`)

## Setup

### 1. Create and activate a virtual environment

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If you already use a project environment (for example the notebook kernel environment), activate that instead.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set the OpenAI API key

Option A: export it in your shell

```bash
export OPENAI_API_KEY="your_key_here"
```

Option B: create a local `.env` file (supported by the script)

```env
OPENAI_API_KEY=your_key_here
```

The updater auto-loads `.env` if present.

## Running the updater

Main command:

```bash
python update_alexandras_data.py --verbose
```

What happens during a run:

- A log file is created under `logs/` by default (timestamped).
- The script fetches Diavgeia search results.
- It compares fetched rows to existing CSV rows.
- It processes only new candidates.
- It downloads missing PDFs and reuses cached PDFs.
- It summarizes with OpenAI and cleans the response JSON.
- It appends relevant rows to `data_alexandras.csv`.

### Recommended first test run

Use a limit to test the pipeline safely:

```bash
python update_alexandras_data.py --limit 3 --verbose
```

## Updater options

### Logging

- Default behavior: create `logs/update_alexandras_YYYYMMDD_HHMMSS.log`
- Custom log file:

```bash
python update_alexandras_data.py --log-file logs/manual_run.log --verbose
```

- Disable log file:

```bash
python update_alexandras_data.py --no-log-file --verbose
```

### Existing summary cleanup (opt-in)

By default, old rows are not touched.

If you explicitly want to normalize/clean existing JSON summaries in `data_alexandras.csv`, run:

```bash
python update_alexandras_data.py --clean-existing --verbose
```

Use this carefully, because it rewrites existing `summary` values (JSON formatting/normalization), even if no new data is found.

### Include irrelevant results

Default behavior: only append rows where the OpenAI summary marks `relevance = "YES"`.

To also append `relevance = "NO"` rows:

```bash
python update_alexandras_data.py --include-irrelevant --verbose
```

### Other useful flags

- `--limit N`: process only first `N` new candidate rows
- `--sleep 0.2`: delay between Diavgeia page requests
- `--model gpt-5-mini`: choose OpenAI model
- `--csv path/to/file.csv`: alternate CSV output path
- `--pdf-dir data/pdfs`: alternate PDF cache directory

## Summary JSON format

The updater asks OpenAI to return strict JSON with these keys:

- `relevance`
- `description`
- `table`
- `companies`
- `pages`
- `extras`

Then the script cleans the result to make it safer and more consistent:

- extracts JSON if the model wraps output with extra text
- removes markdown fences if present
- normalizes `relevance` to `YES` / `NO`
- trims `description` length (word-limited)
- normalizes list fields (`table`, `companies`)
- parses/normalizes `pages` into unique positive integers
- removes duplicates in `table` and `companies`
- clears noisy details for `relevance = NO` rows

## Publishing workflow (GitHub Pages repo)

Use the helper script:

```bash
./publish_alexandras_update.sh
```

This script performs:

1. `python update_alexandras_data.py --log-file logs/manual_run.log --verbose`
2. Copies:
   - `index.html`
   - `data_alexandras.csv`
   to `/Users/troboukis/Code/troboukis.github.io/alexandras`
3. `cd /Users/troboukis/Code/troboukis.github.io`
4. `git add`, `commit`, and `push` only:
   - `alexandras/index.html`
   - `alexandras/data_alexandras.csv`

### Custom commit message

```bash
./publish_alexandras_update.sh "alexandras: refresh page and data"
```

If there are no changes in those two files, the script exits without creating a commit.

## Notes on the notebook (`get_data.ipynb`)

The notebook is useful for exploration and debugging, but the updater script is the preferred way to run repeatable updates because it:

- supports incremental updates
- adds progress logging
- normalizes OpenAI summaries
- preserves existing rows unless explicitly asked to clean them
- is easier to schedule and automate

## Troubleshooting

### `OPENAI_API_KEY is not set`

- Set the variable in your shell, or
- add it to `.env` in the repo root

### `ModuleNotFoundError` (e.g. `requests`, `fitz`, `openai`)

Your current Python interpreter is not using the correct environment.

Fix:

```bash
pip install -r requirements.txt
```

and ensure you run the script with that environment’s Python.

### No new rows appended

Possible reasons:

- Diavgeia returned no new matching decisions
- New decisions were fetched but marked `relevance = NO`
- Fetched rows were exact metadata duplicates of rows already in the CSV

Check the log file in `logs/` for per-document decisions and errors.

## Operational recommendations

- Test with `--limit` before large runs
- Keep `data/pdfs/` cached (saves time and bandwidth)
- Review logs after each run
- Commit/publish only after checking that new rows look correct

## License / usage

No license file is currently included in this repository. Add one if you plan to publish the code for reuse.
