# Copilot / AI Agent Instructions for this repo

Purpose: Short, actionable guidance to help an AI coding agent become immediately productive in this project.

---

## Quick run / developer workflow ✅
# Copilot / AI Agent Instructions — concise, repo-specific

Purpose: Give an AI coding agent the exact knowledge to be productive quickly in this project.

## Quick workflow & fast iteration
- Shell tips: `export HF_TOKEN="<your_hf_token>"` and `export TOKENIZERS_PARALLELISM=false`.
- Reduce runtime for tests: edit `DAYS` (top of pipelines) or `max_articles` in `src/huggingface_news_fetcher.py`.
- Canonical run commands (copyable):
  - `python3 run_analysis.py`
  - `python3 research_backed_pipeline.py 2>&1 | tee logs/research_backed.log`
  - `python3 complete_1999_2025_all_models.py 2>&1 | tee logs/complete_1999_2025.log`

## Big-picture architecture
- Entrypoints: `run_analysis.py` (main orchestration), `research_backed_pipeline.py`, `complete_1999_2025_all_models.py`.
- Core modules live in `src/`:
  - `src/data_preprocessor.py` — `StockDataProcessor`: fetches prices, computes returns, stationarity tests, rolling features.
  - `src/huggingface_news_fetcher.py` — streaming fetches from `data/fnspid_news/*.parquet`.
  - `advanced_sentiment.py` — `MultiMethodSentimentAnalyzer` (TextBlob, Vader, FinBERT).
  - Models: e.g., `src/tcn_model.py`, `src/transfer_learning_model.py`.
- Data flow: fetch news & prices → daily aggregation → compute multi-method sentiment → rolling features (`_RM{window}`) → merge → train/evaluate → save CSVs/plots to `results/`.

## Project conventions you must follow
- Sentiment columns contain `textblob`, `vader`, or `finbert` — use these substrings for selection.
- Rolling feature names use `_RM{window}` (example: `Close_RM7`).
- Results stored under `results/` (subfolders per pipeline). Logs under `logs/`.
- Data caches: `data/fnspid_news/` (parquet), `data/news_articles/` (csv), and `data/historical_cache/`.
- Many scripts include fallback HF tokens (convenience only). Do NOT commit tokens; prefer `HF_TOKEN` or `--hf-token`.

## Integration & external dependencies
- HuggingFace dataset (streaming): `Brianferrell787/financial-news-multisource` via `datasets.load_dataset`.
- Transformers / FinBERT used in `advanced_sentiment.py` — code uses `torch.device('cuda' if available else 'cpu')`.
- Price data fetched with `yfinance` in `src/data_preprocessor.py`.

## Debugging & editing tips (practical)
- For data-processing changes: run the subroutine in a small REPL and inspect `results/enhanced/` or `results/research_backed/` outputs.
- Watch for off-by-one when computing log returns: `compute_log_returns` drops the first index; check `prepare_time_series_data` alignment.
- When changing sentiment code, validate both HuggingFace fetch and RSS fallback paths used in `run_analysis.py`.

## Small reproducible example (dev)
```py
from src.huggingface_news_fetcher import HuggingFaceFinancialNewsDataset
import os
hf = HuggingFaceFinancialNewsDataset(hf_token=os.environ.get('HF_TOKEN'))
df = hf.fetch_news_for_stock('AAPL','2024-01-01','2025-01-01', max_articles=200)
print(df.head())
```

## Where to look first (key files)
- `run_analysis.py` — main pipeline and orchestration
- `src/data_preprocessor.py` — price fetching, returns, rolling features
- `src/huggingface_news_fetcher.py` and `advanced_sentiment.py` — news + sentiment
- `results/` and `logs/` — outputs for quick verification

If you'd like, I can expand any section (e.g., longer architecture notes, run examples, or developer checklists). Please tell me what to add or clarify.
- To validate a change in data processing: run a single-phase script (e.g., `src/data_preprocessor.py` functions in a small REPL snippet) and inspect `results/enhanced/` outputs.
