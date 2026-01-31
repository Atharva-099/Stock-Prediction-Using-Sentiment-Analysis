#!/usr/bin/env python3
"""
NEWS FREQUENCY ANALYSIS
=======================
Counts the number of financial news articles per day
for a given stock over the last 5 years (2020–2025).
Uses Hugging Face dataset streamed remotely if available,
otherwise falls back to Google RSS.

Author: Atharva
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

# Import Hugging Face fetcher
try:
    from src.huggingface_news_fetcher import HuggingFaceFinancialNewsDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Hugging Face fetcher not found. Please install or use fallback method.")

# Try importing fallback Google RSS method
try:
    from advanced_sentiment import compute_multi_method_sentiment
    RSS_AVAILABLE = True
except ImportError:
    RSS_AVAILABLE = False
    print("Google RSS fetcher not found. Please ensure advanced_sentiment is available.")

# Import Hugging Face streaming loader
try:
    from datasets import load_dataset
    HF_STREAM_AVAILABLE = True
except ImportError:
    HF_STREAM_AVAILABLE = False
    print("Hugging Face datasets library not found, streaming disabled.")

# Configuration
TICKER = 'AAPL'
DAYS = 1825  # ~5 years
end_date = datetime.now()
start_date = end_date - timedelta(days=DAYS)
HUGGINGFACE_TOKEN = os.environ.get('HF_TOKEN', 'YOUR_HF_TOKEN')


def ticker_in_extra_fields(extra_json_str, ticker):
    try:
        data = json.loads(extra_json_str)
        stocks = data.get("stocks", [])
        return ticker in stocks or f"${ticker}" in stocks
    except Exception:
        return False


def fetch_extended_hf_news_stream(ticker, max_items=500, token=None):
    """Stream full financial news dataset from Hugging Face,
       filter by ticker, and return DataFrame."""
    if not HF_STREAM_AVAILABLE:
        print("Hugging Face streaming loader unavailable.")
        return pd.DataFrame()
    print("Starting to stream full financial news dataset from Hugging Face...")
    dataset = load_dataset(
        "Brianferrell787/financial-news-multisource",
        split="train",
        streaming=True,
        token=token
    )
    
    # Use manual iteration with progress - NO LIMIT
    from tqdm import tqdm
    articles = []
    scanned = 0
    sample_checked = False  # For debugging
    
    print(f"Scanning for articles about {ticker} (will stop after {max_items} matches)...")
    
    for row in tqdm(dataset, desc="Scanning dataset", total=None):
        scanned += 1
        
        # Early stopping if we have enough articles
        if len(articles) >= max_items:
            print(f"✓ Found {len(articles)} articles!")
            break
        
        # Debug: show sample of extra_fields structure
        if not sample_checked and scanned % 10000 == 0:
            try:
                extra = json.loads(row.get("extra_fields", "{}"))
                print(f"\n[Debug] Sample extra_fields at record {scanned}: {extra}")
                sample_checked = True
            except:
                pass
        
        try:
            # Check if ticker is in extra_fields
            extra_fields_str = row.get("extra_fields", "{}")
            if ticker_in_extra_fields(extra_fields_str, ticker):
                articles.append(row)
                if len(articles) % 100 == 0:
                    print(f"  Found {len(articles)} articles so far...")
        except Exception as e:
            continue
    
    if not articles:
        print(f"[Extended HF news] No data found after scanning {scanned:,} records.")
        print(f"  Consider: The dataset might not have ticker info in extra_fields, or {ticker} articles are very sparse.")
        return pd.DataFrame()
    
    df = pd.DataFrame(articles)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date').reset_index(drop=True)
    print(f"[Extended HF news] Filtered dataset size: {len(df)}")
    print(f"[Extended HF news] Date range: {df['date'].min()} to {df['date'].max()}")
    return df


# ---------------- Hugging Face ----------------
hf_daily = None
if HF_AVAILABLE and HF_STREAM_AVAILABLE:
    articles_df = fetch_extended_hf_news_stream(TICKER, max_items=500, token=HUGGINGFACE_TOKEN)

    if not articles_df.empty:
        hf_daily = articles_df.groupby('date').size().reset_index(name='count')
        hf_daily = hf_daily.sort_values('date')
        print(f"Hugging Face range: {hf_daily['date'].min()} to {hf_daily['date'].max()}")
        print(f"Total HF days: {len(hf_daily)}")
    else:
        print("No articles found for this ticker in the Hugging Face dataset.")
else:
    print("Hugging Face fetcher or streaming not available. Cannot fetch articles.")


# ---------------- Google RSS ----------------
rss_daily = None
if RSS_AVAILABLE:
    rss_df = compute_multi_method_sentiment(
        f"{TICKER} stock",
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        max_items=2000
    )
    if rss_df is not None and not rss_df.empty:
        if 'date' in rss_df.columns:
            rss_daily = rss_df.groupby('date').size().reset_index(name='count')
        elif 'Date' in rss_df.columns:
            rss_daily = rss_df.groupby('Date').size().reset_index(name='count').rename(columns={'Date': 'date'})
        else:
            rss_df['_tmp_date'] = pd.to_datetime(rss_df.get('created_at', pd.Timestamp.now()))
            rss_daily = rss_df.groupby('_tmp_date').size().reset_index(name='count').rename(columns={'_tmp_date': 'date'})
        rss_daily = rss_daily.sort_values('date')
        print(f"Google RSS range: {rss_daily['date'].min()} to {rss_daily['date'].max()}")
        print(f"Total RSS days: {len(rss_daily)}")
    else:
        print("No articles found in the Google RSS dataset.")
else:
    print("Google RSS fetcher not available. Cannot fetch articles.")


# ---------------- Plot results ----------------
if hf_daily is not None:
    plt.figure(figsize=(15,5))
    plt.plot(pd.to_datetime(hf_daily['date']), hf_daily['count'], marker='o', markersize=3, color='skyblue')
    plt.title(f'Hugging Face News Frequency ({TICKER})')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('results/enhanced/statistical', exist_ok=True)
    plt.savefig('results/enhanced/statistical/hf_news_frequency.png', dpi=300)
    plt.show()

if rss_daily is not None:
    plt.figure(figsize=(15,5))
   
    # Convert dates and ensure correct dtype
    rss_daily['date'] = pd.to_datetime(rss_daily['date'])
    rss_daily = rss_daily.set_index('date').asfreq('D', fill_value=0).reset_index()
   
    # Apply rolling window smoothing
    rss_daily['spike'] = rss_daily['count'].rolling(window=3, center=True).mean().fillna(rss_daily['count'])
   
    # Highlight spikes
    spike_threshold = rss_daily['count'].mean() + 1.5 * rss_daily['count'].std()
    spikes = rss_daily[rss_daily['count'] > spike_threshold]
   
    # Plot smoother and spikes
    plt.plot(rss_daily['date'], rss_daily['spike'], color='orange', linewidth=2, label='Smoothed Frequency')
    plt.scatter(spikes['date'], spikes['count'], color='red', s=50, label='Spikes (High Activity)')
   
    plt.title(f'Google RSS News Frequency ({TICKER}) - Enhanced View')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    os.makedirs('results/enhanced/statistical', exist_ok=True)
    plt.savefig('results/enhanced/statistical/rss_news_frequency.png', dpi=300)
    plt.show()