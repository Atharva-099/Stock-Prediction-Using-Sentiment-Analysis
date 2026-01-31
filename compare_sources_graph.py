#!/usr/bin/env python3
"""
Compare Hugging Face vs fallback (Google RSS) article frequency over the selected period.
Saves results/enhanced/statistical/compare_sources_bar.png
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "enhanced", "statistical")
os.makedirs(OUT_DIR, exist_ok=True)

TICKER = "AAPL"
DAYS = 1825
end_date = datetime.now()
start_date = end_date - timedelta(days=DAYS)

# Utility: aggregate a daily-count Series to monthly totals for plotting
def monthly_aggregate(df, date_col="date", count_col="count"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    monthly = df[count_col].resample("M").sum().rename("count")
    monthly.index = monthly.index.to_period("M").to_timestamp()
    return monthly

# 1) Try to load HuggingFace via your fetcher
hf_daily = None
try:
    from src.huggingface_news_fetcher import HuggingFaceFinancialNewsDataset
    print("Fetching Hugging Face data...")
    HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")
    hf_fetcher = HuggingFaceFinancialNewsDataset(hf_token=HUGGINGFACE_TOKEN)
    articles_df = hf_fetcher.fetch_news_for_stock(
        ticker=TICKER,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        max_articles=5000
    )
    if not articles_df.empty:
        try:
            daily = hf_fetcher.aggregate_to_daily(articles_df)
            if "date" in daily.columns:
                daily_count = daily.groupby("date").size().reset_index(name="count")
            else:
                daily_count = articles_df.groupby("date").size().reset_index(name="count")
        except Exception:
            daily_count = articles_df.groupby("date").size().reset_index(name="count")
        hf_daily = daily_count.sort_values("date").reset_index(drop=True)
        print(f"Hug Face range: {hf_daily['date'].min()} to {hf_daily['date'].max()}")
        print(f"Total HF days: {len(hf_daily)}")
    else:
        print("Hugging Face fetcher returned 0 articles.")
except Exception as e:
    print("Hugging Face fetcher not available or failed:", e)

# 2) Try to obtain fallback (Google RSS) data by reusing compute_multi_method_sentiment if present
rss_daily = None
try:
    from advanced_sentiment import compute_multi_method_sentiment
    print("Fetching fallback Google RSS (compute_multi_method_sentiment)...")
    rss_df = compute_multi_method_sentiment(
        f"{TICKER} stock",
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        max_items=2000
    )
    if rss_df is not None and not rss_df.empty:
        if "date" in rss_df.columns:
            rss_daily = rss_df.groupby("date").size().reset_index(name="count")
        elif "Date" in rss_df.columns:
            rss_daily = rss_df.groupby("Date").size().reset_index(name="count")
        else:
            rss_df["_tmp_date"] = pd.to_datetime(rss_df.get("created_at", pd.Timestamp.now()))
            rss_daily = rss_df.groupby("_tmp_date").size().reset_index(name="count").rename(columns={"_tmp_date": "date"})
        rss_daily = rss_daily.sort_values("date").reset_index(drop=True)
        print(f"Fallback RSS days: {len(rss_daily)} (range {rss_daily['date'].min()} to {rss_daily['date'].max()})")
    else:
        print("Fallback RSS fetch returned no rows or fetcher returned empty.")
except Exception as e:
    print("Fallback RSS fetcher not available or failed:", e)

# 3) If neither fetcher is available, try to infer article frequency from your merged CSV created by run_analysis.py
if hf_daily is None and rss_daily is None:
    merged_path = os.path.join(PROJECT_ROOT, "results", "enhanced", "enhanced_dataset_with_all_features.csv")
    if os.path.exists(merged_path):
        print("No fetchers available — reading merged dataset to infer coverage...")
        merged = pd.read_csv(merged_path)
        date_col = None
        for candidate in ("Date", "date", "DATE"):
            if candidate in merged.columns:
                date_col = candidate
                break
        if date_col is None:
            raise ValueError("No date column found in enhanced_dataset_with_all_features.csv — cannot infer article counts.")
        text_candidates = [c for c in merged.columns if any(k in c.lower() for k in ("text", "article", "headline", "full_text"))]
        if text_candidates:
            chosen_text = text_candidates[0]
            merged["_has_text"] = merged[chosen_text].notnull() & (merged[chosen_text].astype(str).str.strip() != "")
            daily = merged.groupby(date_col)["_has_text"].sum().reset_index(name="count")
            daily = daily.sort_values(date_col)
            daily = daily.rename(columns={date_col: "date"})
            hf_daily = daily
            print(f"Inferred data from merged dataset. Days with non-empty text entries: {len(hf_daily)}")
        else:
            sentiment_cols = [c for c in merged.columns if any(k in c.lower() for k in ("textblob", "vader", "finbert"))]
            if sentiment_cols:
                merged["_has_sent"] = merged[sentiment_cols].notnull().any(axis=1)
                daily = merged.groupby(date_col)["_has_sent"].sum().reset_index(name="count")
                daily = daily.sort_values(date_col).rename(columns={date_col: "date"})
                hf_daily = daily
                print(f"Inferred data from sentiment columns in merged dataset. Days: {len(hf_daily)}")
            else:
                raise ValueError("Cannot infer article counts from merged dataset (no text/sentiment columns found).")

# 4) Plot two separate bar graphs for Hugging Face and Google RSS
if hf_daily is None and rss_daily is None:
    raise SystemExit("No data available from any source to plot.")

# Determine date ranges
hf_start = pd.to_datetime(hf_daily["date"].min()) if hf_daily is not None else None
hf_end = pd.to_datetime(hf_daily["date"].max()) if hf_daily is not None else None
rss_start = pd.to_datetime(rss_daily["date"].min()) if rss_daily is not None else None
rss_end = pd.to_datetime(rss_daily["date"].max()) if rss_daily is not None else None

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

# Hugging Face coverage
if hf_start is not None and hf_end is not None:
    duration = (hf_end - hf_start).days
    axes[0].barh(0, duration, left=hf_start.toordinal(), color="skyblue")
    axes[0].set_yticks([0])
    axes[0].set_yticklabels(["Hugging Face"])
    axes[0].set_title(f"{TICKER} — Hugging Face Data Coverage ({hf_start.date()} to {hf_end.date()})")

# Google RSS coverage
if rss_start is not None and rss_end is not None:
    duration = (rss_end - rss_start).days
    axes[1].barh(0, duration, left=rss_start.toordinal(), color="orange")
    axes[1].set_yticks([0])
    axes[1].set_yticklabels(["Google RSS"])
    axes[1].set_title(f"{TICKER} — Google RSS Data Coverage ({rss_start.date()} to {rss_end.date()})")

# Format x-axis as dates
all_starts = [d for d in [hf_start, rss_start] if d is not None]
all_ends = [d for d in [hf_end, rss_end] if d is not None]
ticks = pd.date_range(min(all_starts), max(all_ends), freq="3M")

for ax in axes:
    ax.set_xticks([t.toordinal() for t in ticks])
    ax.set_xticklabels(ticks.strftime("%b %Y"), rotation=45)
    ax.set_xlabel("Date")
    ax.set_xlim(min(all_starts).toordinal(), max(all_ends).toordinal())

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "compare_sources_bar.png")
fig.savefig(out_path, dpi=150)
print("Saved separate coverage graphs to:", out_path)
plt.show()
