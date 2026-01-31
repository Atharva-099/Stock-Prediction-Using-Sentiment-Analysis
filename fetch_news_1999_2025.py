#!/usr/bin/env python3

"""
Fetch Financial News Articles (1999-2025)
==========================================

Fetches 10,000 articles per year from HuggingFace datasets.
Stores all articles in CSV format for pipeline use.

Total target: ~270,000 articles (1999-2025)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import json
import os
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fetch_news.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
HF_TOKEN = 'YOUR_TOKEN_HERE'
START_YEAR = 1999
END_YEAR = 2025
ARTICLES_PER_YEAR = 10000
DATASET_NAME = "Brianferrell787/financial-news-multisource"
OUTPUT_DIR = 'data/news_articles'
OUTPUT_FILE = f'{OUTPUT_DIR}/all_news_1999_2025.csv'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

logger.info("=" * 80)
logger.info("FETCHING FINANCIAL NEWS ARTICLES (1999-2025)")
logger.info("=" * 80)
logger.info(f"Start Year: {START_YEAR}")
logger.info(f"End Year: {END_YEAR}")
logger.info(f"Articles per Year: {ARTICLES_PER_YEAR}")
logger.info(f"Target Total: ~{ARTICLES_PER_YEAR * (END_YEAR - START_YEAR + 1):,} articles")
logger.info(f"Output File: {OUTPUT_FILE}\n")


def fetch_fnspid_news():
    """
    Fetch news from FNSPID dataset (1999-2023)
    This is the most comprehensive historical dataset
    """
    from datasets import load_dataset
    
    logger.info("=" * 80)
    logger.info("STEP 1: Fetching from FNSPID (1999-2023)")
    logger.info("=" * 80)
    
    articles = []
    yearly_counts = defaultdict(int)
    
    try:
        # Load FNSPID dataset
        logger.info("Loading FNSPID dataset from HuggingFace...")
        ds = load_dataset(
            DATASET_NAME,
            data_files="data/fnspid_news/*.parquet",
            split="train",
            streaming=True,
            token=HF_TOKEN
        )
        
        total_target = ARTICLES_PER_YEAR * (END_YEAR - START_YEAR + 1)
        logger.info(f"Streaming dataset (target: {total_target:,} articles)...\n")
        
        pbar = tqdm(ds, desc="FNSPID", total=total_target * 2)
        
        for row in pbar:
            # Check if we have enough articles for all years
            if all(yearly_counts[y] >= ARTICLES_PER_YEAR 
                   for y in range(START_YEAR, END_YEAR + 1)):
                logger.info(f"\n✓ Reached target for all years!")
                break
            
            try:
                # Parse date
                article_date = pd.to_datetime(row['date'])
                year = article_date.year
                
                # Skip if outside our range
                if year < START_YEAR or year > END_YEAR:
                    continue
                
                # Skip if we already have enough for this year
                if yearly_counts[year] >= ARTICLES_PER_YEAR:
                    continue
                
                # Extract text
                text = row.get('text', '').strip()
                if not text or len(text) < 10:
                    continue
                
                # Parse extra fields
                extra = {}
                try:
                    extra = json.loads(row.get('extra_fields', '{}'))
                except:
                    pass
                
                # Store article
                articles.append({
                    'date': article_date.date(),
                    'datetime': article_date,
                    'year': year,
                    'month': article_date.month,
                    'text': text[:3000],  # Truncate to 3000 chars
                    'full_text': text,
                    'source': extra.get('source', 'FNSPID'),
                    'ticker': extra.get('ticker', None),
                    'dataset': 'fnspid'
                })
                
                yearly_counts[year] += 1
                pbar.set_postfix({'articles': len(articles), 'years': len(yearly_counts)})
                
            except Exception as e:
                continue
        
        pbar.close()
        
    except Exception as e:
        logger.error(f"Error loading FNSPID: {e}")
        return pd.DataFrame()
    
    # Log summary
    logger.info("\nFNSPID Summary:")
    logger.info("-" * 60)
    total_fnspid = 0
    for year in sorted(yearly_counts.keys()):
        logger.info(f"  {year}: {yearly_counts[year]:,} articles")
        total_fnspid += yearly_counts[year]
    
    logger.info(f"\nTotal from FNSPID: {total_fnspid:,} articles")
    
    return pd.DataFrame(articles)


def fetch_additional_datasets():
    """
    Fetch from additional sources for recent years (2023-2025)
    Uses ashraq/financial-news dataset which has more recent data
    """
    from datasets import load_dataset
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Fetching additional data for recent years (2023-2025)")
    logger.info("=" * 80)
    
    articles = []
    yearly_counts = defaultdict(int)
    
    try:
        logger.info("Loading ashraq/financial-news dataset...")
        ds = load_dataset(
            "ashraq/financial-news",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        logger.info(f"Streaming ashraq dataset (target: 30,000 articles for 2023-2025)...\n")
        
        pbar = tqdm(ds, desc="ashraq", total=100000)
        
        for row in pbar:
            # Only need 2023-2025
            if all(yearly_counts[y] >= ARTICLES_PER_YEAR 
                   for y in range(2023, END_YEAR + 1)):
                logger.info(f"\n✓ Reached target for 2023-2025!")
                break
            
            try:
                # Get date
                date_str = row.get('date') or row.get('published_date')
                if not date_str:
                    continue
                
                article_date = pd.to_datetime(date_str)
                year = article_date.year
                
                # Only fetch 2023-2025
                if year < 2023 or year > END_YEAR:
                    continue
                
                # Skip if we already have enough for this year
                if yearly_counts[year] >= ARTICLES_PER_YEAR:
                    continue
                
                # Get text
                text = row.get('article') or row.get('content') or row.get('headline') or row.get('title', '')
                text = text.strip() if text else ''
                
                if not text or len(text) < 10:
                    continue
                
                # Store article
                articles.append({
                    'date': article_date.date(),
                    'datetime': article_date,
                    'year': year,
                    'month': article_date.month,
                    'text': text[:3000],
                    'full_text': text,
                    'source': row.get('source', 'ashraq'),
                    'ticker': row.get('stock') or row.get('symbol'),
                    'dataset': 'ashraq'
                })
                
                yearly_counts[year] += 1
                pbar.set_postfix({'articles': len(articles), 'years': len(yearly_counts)})
                
            except Exception as e:
                continue
        
        pbar.close()
        
    except Exception as e:
        logger.warning(f"Could not load ashraq dataset: {e}")
        logger.info("Continuing with FNSPID data only...")
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


def combine_and_save_articles(fnspid_df, additional_df):
    """
    Combine articles from all sources and save to CSV
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Combining and Saving Articles")
    logger.info("=" * 80)
    
    # Combine datasets
    if not additional_df.empty:
        combined_df = pd.concat([fnspid_df, additional_df], ignore_index=True)
    else:
        combined_df = fnspid_df.copy()
    
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
    
    stats_file = f'{OUTPUT_DIR}/news_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✓ Saved statistics to: {stats_file}")
    
    return combined_df


def main():
    """Main pipeline"""
    start_time = datetime.now()
    
    try:
        # Step 1: Fetch FNSPID (1999-2023)
        fnspid_df = fetch_fnspid_news()
        
        # Step 2: Fetch additional data (2023-2025)
        additional_df = fetch_additional_datasets()
        
        # Step 3: Combine and save
        final_df = combine_and_save_articles(fnspid_df, additional_df)
        
        # Final summary
        elapsed = datetime.now() - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ NEWS FETCHING PIPELINE COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\n📊 Final Statistics:")
        logger.info(f"   Total articles: {len(final_df):,}")
        logger.info(f"   Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        logger.info(f"   Years covered: {final_df['year'].nunique()}")
        logger.info(f"   CSV file: {OUTPUT_FILE}")
        logger.info(f"\n⏱️  Total time: {elapsed}")
        
        print(f"\n🎉 Success! Fetched {len(final_df):,} news articles")
        print(f"📁 Saved to: {OUTPUT_FILE}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
