#!/usr/bin/env python3

"""
Fetch Recent Financial News Articles (2020-2025)
===============================================

FIXED VERSION - Uses correct ashraq field names:
- date field: 'date'
- text field: 'headline'
- ticker field: 'stock'
- source field: 'publisher'

Note: ashraq dataset contains 2020-2025 data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import json
import os
from tqdm import tqdm
import feedparser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fetch_recent_news.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
HF_TOKEN = 'YOUR_TOKER_HERE'  # HuggingFace token if needed
START_DATE = '2020-01-01'  # ashraq only has 2020+ data
END_DATE = datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = 'data/news_articles'
OUTPUT_FILE = f'{OUTPUT_DIR}/news_2020_2025.csv'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

logger.info("=" * 80)
logger.info("FETCHING RECENT FINANCIAL NEWS ARTICLES (2020-2025)")
logger.info("=" * 80)
logger.info(f"Start Date: {START_DATE}")
logger.info(f"End Date: {END_DATE}")
logger.info(f"Output File: {OUTPUT_FILE}\n")


def fetch_ashraq_news():
    """
    Fetch news from ashraq/financial-news dataset (2020-2025)
    
    CORRECTED FIELD NAMES:
    - date: actual date field
    - headline: news text
    - publisher: source
    - stock: ticker symbol
    """
    from datasets import load_dataset
    
    logger.info("=" * 80)
    logger.info("STEP 1: Fetching from ashraq/financial-news (2020-2025)")
    logger.info("=" * 80)
    
    articles = []
    yearly_counts = defaultdict(int)
    
    try:
        logger.info("Loading ashraq/financial-news dataset...")
        ds = load_dataset(
            "ashraq/financial-news",
            split="train",
            streaming=True
        )
        
        start_dt = pd.to_datetime(START_DATE)
        end_dt = pd.to_datetime(END_DATE)
        
        logger.info(f"Streaming dataset (filtering for {START_DATE} to {END_DATE})...\n")
        
        pbar = tqdm(ds, desc="ashraq", total=500000)
        
        for row in pbar:
            try:
                # Parse date - ashraq uses 'date' field
                date_str = row.get('date')
                if not date_str:
                    continue
                
                try:
                    article_date = pd.to_datetime(date_str)
                except:
                    continue
                
                year = article_date.year
                
                # Only fetch 2020-2025
                if article_date < start_dt or article_date > end_dt:
                    continue
                
                # Extract text - ashraq uses 'headline' field
                text = row.get('headline')
                if not text:
                    continue
                
                text = text.strip() if text else ''
                
                # Skip very short texts
                if not text or len(text) < 10:
                    continue
                
                # Get source - ashraq uses 'publisher' field
                source = row.get('publisher', 'ashraq')
                
                # Get ticker - ashraq uses 'stock' field
                ticker = row.get('stock')
                
                # Store article
                articles.append({
                    'date': article_date.date(),
                    'datetime': article_date,
                    'year': year,
                    'month': article_date.month,
                    'text': text[:3000],  # Truncate to 3000 chars
                    'full_text': text,
                    'source': source,
                    'ticker': ticker,
                    'dataset': 'ashraq'
                })
                
                yearly_counts[year] += 1
                pbar.set_postfix({'articles': len(articles), 'years': len(yearly_counts)})
                
            except Exception as e:
                continue
        
        pbar.close()
        
    except Exception as e:
        logger.warning(f"Error loading ashraq: {e}")
        logger.info("Will proceed with RSS feeds...")
        return pd.DataFrame()
    
    # Log summary
    logger.info("\nashraq Summary:")
    logger.info("-" * 60)
    total_ashraq = 0
    for year in sorted(yearly_counts.keys()):
        logger.info(f"  {year}: {yearly_counts[year]:,} articles")
        total_ashraq += yearly_counts[year]
    
    logger.info(f"\nTotal from ashraq: {total_ashraq:,} articles")
    
    return pd.DataFrame(articles)


def fetch_rss_news():
    """
    Fetch recent news from RSS feeds
    Backup source for very recent data
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Fetching from RSS Feeds (Backup)")
    logger.info("=" * 80)
    
    articles = []
    
    # Multiple RSS feed sources
    rss_sources = {
        'Google Finance': 'https://news.google.com/rss/topics/CAAqKggKFHRvcF9iZXN0X3N0b2NrcxCNh6MBFgoKCEEwMlZGN0pjKkIKCEEwMlZGN0pj?hl=en-US&gl=US&ceid=US:en',
        'CNBC Top Stories': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
        'Bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
        'Yahoo Finance': 'https://feeds.finance.yahoo.com/rss/topstories.xml'
    }
    
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    
    logger.info(f"Fetching from {len(rss_sources)} RSS sources...")
    
    for source_name, url in rss_sources.items():
        try:
            logger.info(f"\nFetching from {source_name}...")
            feed = feedparser.parse(url)
            
            source_count = 0
            
            for entry in feed.entries:
                try:
                    # Get publication date
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])
                    
                    if not pub_date:
                        continue
                    
                    # Check if within date range
                    if pub_date < start_dt or pub_date > end_dt:
                        continue
                    
                    # Get text
                    text = entry.get('title', '')
                    if hasattr(entry, 'summary'):
                        text = f"{text} {entry.summary}"
                    
                    text = text.strip()
                    if not text or len(text) < 10:
                        continue
                    
                    # Store article
                    articles.append({
                        'date': pub_date.date(),
                        'datetime': pub_date,
                        'year': pub_date.year,
                        'month': pub_date.month,
                        'text': text[:3000],
                        'full_text': text,
                        'source': source_name,
                        'ticker': None,
                        'dataset': 'rss'
                    })
                    
                    source_count += 1
                    
                except Exception as e:
                    continue
            
            logger.info(f"  ✓ {source_count} articles from {source_name}")
            
        except Exception as e:
            logger.warning(f"  ✗ Error fetching {source_name}: {e}")
            continue
    
    # Log summary
    logger.info("\nRSS Summary:")
    logger.info("-" * 60)
    logger.info(f"Total from RSS: {len(articles):,} articles")
    
    return pd.DataFrame(articles)


def combine_and_save_recent_articles(ashraq_df, rss_df):
    """
    Combine articles from all recent sources and save to CSV
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Combining and Saving Articles")
    logger.info("=" * 80)
    
    # Combine all datasets
    dfs_to_combine = []
    if not ashraq_df.empty:
        dfs_to_combine.append(ashraq_df)
    if not rss_df.empty:
        dfs_to_combine.append(rss_df)
    
    if not dfs_to_combine:
        logger.error("No articles collected from any source!")
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs_to_combine, ignore_index=True)
    
    # Remove duplicates
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['date', 'text'], keep='first')
    duplicates_removed = initial_count - len(combined_df)
    
    # Sort by date
    combined_df = combined_df.sort_values('date')
    
    # Log statistics
    logger.info(f"\nCombination Summary:")
    logger.info("-" * 60)
    logger.info(f"Total articles: {len(combined_df):,}")
    logger.info(f"Duplicates removed: {duplicates_removed:,}")
    logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    logger.info(f"Unique dates: {combined_df['date'].nunique():,}")
    
    # Yearly distribution
    logger.info(f"\nYearly Distribution:")
    logger.info("-" * 60)
    yearly_dist = combined_df.groupby('year').size()
    for year in sorted(yearly_dist.index):
        logger.info(f"  {year}: {yearly_dist[year]:,} articles")
    
    # Source distribution
    logger.info(f"\nSource Distribution:")
    logger.info("-" * 60)
    source_dist = combined_df.groupby('source').size()
    for source in sorted(source_dist.index):
        logger.info(f"  {source}: {source_dist[source]:,} articles")
    
    # Save to CSV
    logger.info(f"\nSaving to CSV...")
    combined_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    logger.info(f"✓ Saved {len(combined_df):,} articles to: {OUTPUT_FILE}")
    
    # Save summary stats
    stats = {
        'total_articles': len(combined_df),
        'date_range': {
            'start': str(combined_df['date'].min()),
            'end': str(combined_df['date'].max())
        },
        'unique_dates': int(combined_df['date'].nunique()),
        'yearly_distribution': yearly_dist.to_dict(),
        'source_distribution': source_dist.to_dict()
    }
    
    stats_file = f'{OUTPUT_DIR}/news_stats_recent.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✓ Saved statistics to: {stats_file}")
    
    return combined_df


def combine_csv_files():
    """
    Combine 1999-2024 and 2020-2025 CSV files into one master CSV
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Combining with Historical Data")
    logger.info("=" * 80)
    
    old_csv = f'{OUTPUT_DIR}/all_news_1999_2025.csv'
    new_csv = OUTPUT_FILE
    combined_csv = f'{OUTPUT_DIR}/all_news_combined_final.csv'
    
    try:
        # Load both CSVs
        if os.path.exists(old_csv):
            logger.info(f"Loading historical data from {old_csv}...")
            historical_df = pd.read_csv(old_csv)
            logger.info(f"  ✓ Loaded {len(historical_df):,} historical articles")
        else:
            logger.warning(f"Historical CSV not found: {old_csv}")
            historical_df = pd.DataFrame()
        
        logger.info(f"Loading recent data from {new_csv}...")
        recent_df = pd.read_csv(new_csv)
        logger.info(f"  ✓ Loaded {len(recent_df):,} recent articles")
        
        # Combine
        if not historical_df.empty:
            all_df = pd.concat([historical_df, recent_df], ignore_index=True)
        else:
            all_df = recent_df.copy()
        
        # Remove duplicates
        initial = len(all_df)
        all_df = all_df.drop_duplicates(subset=['date', 'text'], keep='first')
        removed = initial - len(all_df)
        
        # Sort
        all_df = all_df.sort_values('date')
        
        # Convert date column to datetime for consistency
        all_df['date'] = pd.to_datetime(all_df['date']).dt.date
        
        # Save combined
        all_df.to_csv(combined_csv, index=False, encoding='utf-8')
        
        logger.info(f"\n✓ Combined dataset: {len(all_df):,} articles")
        logger.info(f"✓ Duplicates removed: {removed:,}")
        logger.info(f"✓ Date range: {all_df['date'].min()} to {all_df['date'].max()}")
        logger.info(f"✓ Saved to: {combined_csv}")
        
        return all_df
        
    except Exception as e:
        logger.error(f"Error combining CSVs: {e}")
        return None


def main():
    """Main pipeline"""
    start_time = datetime.now()
    
    try:
        # Step 1: Fetch ashraq
        ashraq_df = fetch_ashraq_news()
        
        # Step 2: Fetch RSS feeds
        rss_df = fetch_rss_news()
        
        # Step 3: Combine and save
        final_df = combine_and_save_recent_articles(ashraq_df, rss_df)
        
        # Step 4: Combine with historical data
        if not final_df.empty:
            combined_all = combine_csv_files()
        
        # Final summary
        elapsed = datetime.now() - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ RECENT NEWS FETCHING PIPELINE COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\n📊 Final Statistics:")
        logger.info(f"   Total recent articles (2020-2025): {len(final_df):,}")
        logger.info(f"   Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        logger.info(f"   CSV file: {OUTPUT_FILE}")
        logger.info(f"\n📁 Files Created:")
        logger.info(f"   • {OUTPUT_FILE}")
        logger.info(f"   • {OUTPUT_DIR}/all_news_combined_final.csv")
        logger.info(f"\n⏱️  Total time: {elapsed}")
        
        print(f"\n🎉 Success! Fetched {len(final_df):,} recent articles (2020-2025)")
        print(f"📁 Saved to: {OUTPUT_FILE}")
        print(f"📁 Combined file: {OUTPUT_DIR}/all_news_combined_final.csv")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
