#!/usr/bin/env python3
"""
FETCH MISSING YEARS DATA
=========================
Fetches data for years not covered by the main pipeline:
- 1999-2008: From official FNSPID (Zihan1004/FNSPID)
- 2021-2025: From official FNSPID + RSS feeds

This complements the main pipeline which covers 2009-2020.

Author: CMU Financial Forecasting Project
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# Setup
os.makedirs('data/fnspid_official', exist_ok=True)
os.makedirs('results/combined_data', exist_ok=True)
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fetch_missing_years.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from src.fnspid_official_fetcher import OfficialFNSPIDFetcher, CombinedDataFetcher

HF_TOKEN = os.environ.get('HF_TOKEN', 'YOUR_HF_TOKEN')


def main():
    logger.info("=" * 100)
    logger.info(" FETCHING MISSING YEARS DATA (1999-2008 + 2021-2025)")
    logger.info("=" * 100)
    logger.info("""
    Current coverage from main pipeline:
      • 2009-2020: FNSPID (Brianferrell787) + ashraq/financial-news
    
    Missing years to fetch:
      • 1999-2008: Official FNSPID (Zihan1004/FNSPID)
      • 2021-2023: Official FNSPID (Zihan1004/FNSPID)
      • 2024-2025: RSS Feeds
    """)
    
    # Initialize fetcher
    fnspid_fetcher = OfficialFNSPIDFetcher(cache_dir='data/fnspid_official')
    
    # =========================================================================
    # PART 1: Fetch 1999-2008 (older historical data)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PART 1: FETCHING 1999-2008 DATA")
    logger.info("=" * 80)
    
    older_years = list(range(1999, 2009))
    logger.info(f"Years to fetch: {older_years}")
    
    older_data = fnspid_fetcher.fetch_news_for_years(
        years_needed=older_years,
        max_per_year=30000
    )
    
    if not older_data.empty:
        logger.info(f"✓ Fetched {len(older_data):,} articles for 1999-2008")
        older_data.to_parquet('data/fnspid_official/news_1999_2008.parquet')
        logger.info("✓ Saved: data/fnspid_official/news_1999_2008.parquet")
    else:
        logger.warning("Could not fetch 1999-2008 data")
    
    # =========================================================================
    # PART 2: Fetch 2021-2023 (recent historical data)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PART 2: FETCHING 2021-2023 DATA")
    logger.info("=" * 80)
    
    recent_years = [2021, 2022, 2023]
    logger.info(f"Years to fetch: {recent_years}")
    
    recent_data = fnspid_fetcher.fetch_news_for_years(
        years_needed=recent_years,
        max_per_year=30000
    )
    
    if not recent_data.empty:
        logger.info(f"✓ Fetched {len(recent_data):,} articles for 2021-2023")
        recent_data.to_parquet('data/fnspid_official/news_2021_2023.parquet')
        logger.info("✓ Saved: data/fnspid_official/news_2021_2023.parquet")
    else:
        logger.warning("Could not fetch 2021-2023 data")
    
    # =========================================================================
    # PART 3: Fetch 2024-2025 from RSS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PART 3: FETCHING 2024-2025 DATA (RSS)")
    logger.info("=" * 80)
    
    from src.historical_data_fetcher import RecentDataFetcher
    
    rss_fetcher = RecentDataFetcher(hf_token=HF_TOKEN)
    rss_data = rss_fetcher.fetch_rss_news(query="stock market financial", days_back=400)
    
    if not rss_data.empty:
        rss_data['year'] = pd.to_datetime(rss_data['date']).dt.year
        rss_2024_2025 = rss_data[rss_data['year'] >= 2024]
        
        logger.info(f"✓ Fetched {len(rss_2024_2025):,} articles for 2024-2025")
        rss_2024_2025.to_parquet('data/fnspid_official/news_2024_2025_rss.parquet')
        logger.info("✓ Saved: data/fnspid_official/news_2024_2025_rss.parquet")
    else:
        logger.warning("Could not fetch 2024-2025 RSS data")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("FETCH SUMMARY")
    logger.info("=" * 100)
    
    total_articles = 0
    year_summary = {}
    
    if not older_data.empty:
        for year in older_data['year'].unique():
            count = len(older_data[older_data['year'] == year])
            year_summary[year] = ('FNSPID_Official', count)
            total_articles += count
    
    if not recent_data.empty:
        for year in recent_data['year'].unique():
            count = len(recent_data[recent_data['year'] == year])
            year_summary[year] = ('FNSPID_Official', count)
            total_articles += count
    
    if not rss_data.empty and not rss_2024_2025.empty:
        for year in rss_2024_2025['year'].unique():
            count = len(rss_2024_2025[rss_2024_2025['year'] == year])
            year_summary[year] = ('RSS', count)
            total_articles += count
    
    logger.info(f"\nTotal new articles fetched: {total_articles:,}")
    logger.info("\nYear-by-year breakdown:")
    for year in sorted(year_summary.keys()):
        source, count = year_summary[year]
        logger.info(f"  {year}: {count:,} articles from {source}")
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE DATA COVERAGE (Combined)")
    logger.info("=" * 60)
    logger.info("""
    ✓ 1999-2008: FNSPID Official (Zihan1004/FNSPID)
    ✓ 2009-2020: FNSPID Multi-source (Brianferrell787) + ashraq
    ✓ 2021-2023: FNSPID Official (Zihan1004/FNSPID)  
    ✓ 2024-2025: RSS Feeds (Google, Yahoo, CNBC, MarketWatch)
    
    Total coverage: 1999-2025 (26 years!)
    """)
    
    logger.info("\n📁 OUTPUT FILES:")
    logger.info("  • data/fnspid_official/news_1999_2008.parquet")
    logger.info("  • data/fnspid_official/news_2021_2023.parquet")
    logger.info("  • data/fnspid_official/news_2024_2025_rss.parquet")
    
    print("\n🎉 Missing years data fetched! Now re-run train_historical_model.py for full coverage.")


if __name__ == "__main__":
    main()


