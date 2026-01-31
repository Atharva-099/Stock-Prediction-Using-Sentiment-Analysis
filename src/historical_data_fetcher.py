"""
Historical Financial News Data Fetcher
========================================
Fetches maximum historical data from HuggingFace datasets:
- FNSPID: 1999-2023 (15.7M articles)
- Multi-source: Aggregated from multiple sources

Provides year-by-year data for training historical models.

Author: CMU Financial Forecasting Project
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import json
import os
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)


class HistoricalFinancialDataFetcher:
    """
    Fetches historical financial news data from HuggingFace datasets.
    Optimized for training on long time horizons (1999-2023).
    """
    
    def __init__(self, hf_token=None, cache_dir='data/historical_cache'):
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.dataset_name = "Brianferrell787/financial-news-multisource"
        
        # Available subsets with their date ranges
        self.available_subsets = {
            'fnspid': {
                'files': 'data/fnspid_news/*.parquet',
                'date_range': (1999, 2023),
                'description': 'FNSPID - Financial News and Stock Price Integration Dataset'
            },
            'reuters': {
                'files': 'data/reuters_news/*.parquet',
                'date_range': (2006, 2020),
                'description': 'Reuters Financial News'
            },
            'cnbc': {
                'files': 'data/cnbc_headlines/*.parquet',
                'date_range': (2010, 2023),
                'description': 'CNBC Headlines'
            },
            'seeking_alpha': {
                'files': 'data/seeking_alpha/*.parquet',
                'date_range': (2010, 2020),
                'description': 'Seeking Alpha Articles'
            },
            'benzinga': {
                'files': 'data/benzinga/*.parquet',
                'date_range': (2015, 2023),
                'description': 'Benzinga News'
            }
        }
    
    def _get_cache_path(self, subset, start_year, end_year):
        """Get cache file path for a data subset"""
        return os.path.join(self.cache_dir, f"{subset}_{start_year}_{end_year}.parquet")
    
    def fetch_historical_data(self, 
                             start_year=1999, 
                             end_year=2023,
                             subset='fnspid',
                             max_articles_per_year=50000,
                             use_cache=True,
                             tickers=None):
        """
        Fetch historical financial news data for a date range.
        
        Args:
            start_year: Starting year (default 1999)
            end_year: Ending year (default 2023)
            subset: Which dataset subset to use ('fnspid', 'reuters', etc.)
            max_articles_per_year: Maximum articles to fetch per year
            use_cache: Whether to use cached data if available
            tickers: List of tickers to filter for (None = all)
            
        Returns:
            DataFrame with columns: date, text, source, ticker (if available)
        """
        from datasets import load_dataset
        
        cache_path = self._get_cache_path(subset, start_year, end_year)
        
        # Check cache
        if use_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)
        
        if subset not in self.available_subsets:
            logger.error(f"Unknown subset: {subset}. Available: {list(self.available_subsets.keys())}")
            return pd.DataFrame()
        
        subset_info = self.available_subsets[subset]
        data_files = subset_info['files']
        
        logger.info("=" * 80)
        logger.info(f"FETCHING HISTORICAL DATA: {subset_info['description']}")
        logger.info("=" * 80)
        logger.info(f"Date Range: {start_year} - {end_year}")
        logger.info(f"Data Files: {data_files}")
        
        articles = []
        yearly_counts = defaultdict(int)
        
        try:
            ds = load_dataset(
                self.dataset_name,
                data_files=data_files,
                split="train",
                streaming=True,
                token=self.hf_token
            )
            
            total_target = max_articles_per_year * (end_year - start_year + 1)
            
            logger.info(f"Streaming dataset (target: {total_target:,} articles)...")
            
            pbar = tqdm(ds, desc="Fetching", total=total_target)
            
            for row in pbar:
                try:
                    article_date = pd.to_datetime(row['date'])
                    year = article_date.year
                    
                    # Skip if outside date range
                    if year < start_year or year > end_year:
                        continue
                    
                    # Skip if we already have enough for this year
                    if yearly_counts[year] >= max_articles_per_year:
                        # Check if we have enough for all years
                        if all(yearly_counts[y] >= max_articles_per_year 
                               for y in range(start_year, end_year + 1)):
                            break
                        continue
                    
                    # Extract fields
                    text = row.get('text', '')
                    
                    # Parse extra fields
                    extra = {}
                    try:
                        extra = json.loads(row.get('extra_fields', '{}'))
                    except:
                        pass
                    
                    source = extra.get('source', subset)
                    ticker = extra.get('ticker', None)
                    
                    # Filter by ticker if specified
                    if tickers and ticker and ticker not in tickers:
                        continue
                    
                    articles.append({
                        'date': article_date.date(),
                        'datetime': article_date,
                        'year': year,
                        'month': article_date.month,
                        'text': text[:3000],  # Truncate very long texts
                        'full_text': text,
                        'source': source,
                        'ticker': ticker,
                        'dataset': extra.get('dataset', subset)
                    })
                    
                    yearly_counts[year] += 1
                    pbar.set_postfix({
                        'articles': len(articles),
                        'years': len(yearly_counts)
                    })
                    
                except Exception as e:
                    continue
            
            pbar.close()
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
        
        if not articles:
            logger.warning("No articles found!")
            return pd.DataFrame()
        
        df = pd.DataFrame(articles)
        
        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("FETCH SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Articles: {len(df):,}")
        logger.info(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Unique Dates: {df['date'].nunique():,}")
        logger.info("\nYearly Distribution:")
        for year in sorted(yearly_counts.keys()):
            logger.info(f"  {year}: {yearly_counts[year]:,}")
        
        # Cache the data
        if use_cache:
            df.to_parquet(cache_path)
            logger.info(f"\n✓ Cached to: {cache_path}")
        
        return df
    
    def fetch_all_available_data(self, 
                                 start_year=1999, 
                                 end_year=2023,
                                 max_articles_per_year=30000,
                                 combine=True):
        """
        Fetch data from all available subsets and optionally combine them.
        
        Returns:
            If combine=True: Single DataFrame with all data
            If combine=False: Dict of DataFrames by subset
        """
        all_data = {}
        
        for subset_name, subset_info in self.available_subsets.items():
            subset_start = max(start_year, subset_info['date_range'][0])
            subset_end = min(end_year, subset_info['date_range'][1])
            
            if subset_start > subset_end:
                logger.info(f"Skipping {subset_name}: no overlap with {start_year}-{end_year}")
                continue
            
            logger.info(f"\nFetching from {subset_name} ({subset_start}-{subset_end})...")
            
            df = self.fetch_historical_data(
                start_year=subset_start,
                end_year=subset_end,
                subset=subset_name,
                max_articles_per_year=max_articles_per_year
            )
            
            if not df.empty:
                all_data[subset_name] = df
        
        if combine and all_data:
            combined = pd.concat(all_data.values(), ignore_index=True)
            combined = combined.drop_duplicates(subset=['date', 'text'], keep='first')
            combined = combined.sort_values('date')
            logger.info(f"\n✓ Combined dataset: {len(combined):,} articles")
            return combined
        
        return all_data
    
    def get_data_for_training_split(self, 
                                    train_years=(1999, 2020),
                                    val_years=(2021, 2022),
                                    test_years=(2023, 2023),
                                    max_articles_per_year=30000):
        """
        Get data already split into train/val/test sets by year.
        
        Args:
            train_years: (start, end) years for training
            val_years: (start, end) years for validation
            test_years: (start, end) years for testing
            
        Returns:
            dict with 'train', 'val', 'test' DataFrames
        """
        logger.info("=" * 80)
        logger.info("PREPARING DATA SPLITS BY YEAR")
        logger.info("=" * 80)
        logger.info(f"Train: {train_years[0]}-{train_years[1]}")
        logger.info(f"Val:   {val_years[0]}-{val_years[1]}")
        logger.info(f"Test:  {test_years[0]}-{test_years[1]}")
        
        # Fetch all data
        all_years = range(train_years[0], test_years[1] + 1)
        
        all_data = self.fetch_historical_data(
            start_year=train_years[0],
            end_year=test_years[1],
            max_articles_per_year=max_articles_per_year
        )
        
        if all_data.empty:
            return {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
        
        # Split by year
        train_data = all_data[
            (all_data['year'] >= train_years[0]) & 
            (all_data['year'] <= train_years[1])
        ]
        
        val_data = all_data[
            (all_data['year'] >= val_years[0]) & 
            (all_data['year'] <= val_years[1])
        ]
        
        test_data = all_data[
            (all_data['year'] >= test_years[0]) & 
            (all_data['year'] <= test_years[1])
        ]
        
        logger.info("\n" + "=" * 60)
        logger.info("SPLIT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Train: {len(train_data):,} articles ({train_years[0]}-{train_years[1]})")
        logger.info(f"Val:   {len(val_data):,} articles ({val_years[0]}-{val_years[1]})")
        logger.info(f"Test:  {len(test_data):,} articles ({test_years[0]}-{test_years[1]})")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }


class RecentDataFetcher:
    """
    Fetches recent financial news data (2023-2025) from:
    - ashraq/financial-news dataset
    - Google RSS feeds
    """
    
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
    
    def fetch_ashraq_news(self, start_date=None, end_date=None, max_articles=50000):
        """
        Fetch from ashraq/financial-news dataset (2020-2025)
        """
        from datasets import load_dataset
        
        logger.info("=" * 80)
        logger.info("FETCHING RECENT DATA: ashraq/financial-news")
        logger.info("=" * 80)
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
        else:
            start_dt = pd.to_datetime('2023-01-01')
            
        if end_date:
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = datetime.now()
        
        articles = []
        
        try:
            ds = load_dataset(
                "ashraq/financial-news",
                split="train",
                streaming=True
            )
            
            logger.info(f"Filtering for {start_dt.date()} to {end_dt.date()}...")
            
            for row in tqdm(ds, desc="ashraq/financial-news", total=max_articles * 2):
                if len(articles) >= max_articles:
                    break
                    
                try:
                    # Get date
                    date_str = row.get('date') or row.get('published_date')
                    if not date_str:
                        continue
                        
                    article_date = pd.to_datetime(date_str)
                    
                    if article_date < start_dt or article_date > end_dt:
                        continue
                    
                    articles.append({
                        'date': article_date.date(),
                        'datetime': article_date,
                        'year': article_date.year,
                        'text': row.get('headline', '') or row.get('title', ''),
                        'full_text': row.get('article', '') or row.get('content', ''),
                        'source': row.get('source', 'ashraq'),
                        'ticker': row.get('stock', None) or row.get('symbol', None)
                    })
                except:
                    continue
            
        except Exception as e:
            logger.error(f"Error fetching ashraq data: {e}")
            return pd.DataFrame()
        
        df = pd.DataFrame(articles)
        
        if not df.empty:
            logger.info(f"✓ Fetched {len(df):,} articles")
            logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def fetch_rss_news(self, query="stock market", days_back=30):
        """
        Fetch recent news from RSS feeds
        """
        import feedparser
        
        logger.info("=" * 80)
        logger.info("FETCHING RSS NEWS")
        logger.info("=" * 80)
        
        sources = {
            'Google Finance': f'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en',
            'CNBC': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147',
            'MarketWatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
        }
        
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for source_name, url in sources.items():
            try:
                logger.info(f"Fetching from {source_name}...")
                feed = feedparser.parse(url)
                
                for entry in feed.entries:
                    try:
                        pub_date = pd.to_datetime(entry.published)
                        
                        if pub_date < cutoff_date:
                            continue
                            
                        articles.append({
                            'date': pub_date.date(),
                            'datetime': pub_date,
                            'year': pub_date.year,
                            'text': entry.title,
                            'full_text': entry.get('summary', entry.title),
                            'source': source_name,
                            'ticker': None
                        })
                    except:
                        continue
                        
                logger.info(f"  ✓ {len([a for a in articles if a['source'] == source_name])} articles")
                
            except Exception as e:
                logger.warning(f"  ✗ Error: {e}")
        
        df = pd.DataFrame(articles)
        
        if not df.empty:
            df = df.drop_duplicates(subset=['date', 'text'], keep='first')
            logger.info(f"\n✓ Total RSS articles: {len(df)}")
        
        return df
    
    def fetch_all_recent_data(self, start_date='2023-01-01', end_date=None):
        """
        Fetch all available recent data from multiple sources
        """
        all_dfs = []
        
        # ashraq dataset
        ashraq_df = self.fetch_ashraq_news(start_date, end_date)
        if not ashraq_df.empty:
            all_dfs.append(ashraq_df)
        
        # RSS feeds
        rss_df = self.fetch_rss_news(days_back=90)
        if not rss_df.empty:
            all_dfs.append(rss_df)
        
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined = combined.drop_duplicates(subset=['date', 'text'], keep='first')
            combined = combined.sort_values('date')
            logger.info(f"\n✓ Combined recent data: {len(combined):,} articles")
            return combined
        
        return pd.DataFrame()


if __name__ == "__main__":
    import os
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    hf_token = os.environ.get('HF_TOKEN', 'YOUR_HF_TOKEN')
    
    # Test historical fetcher
    historical = HistoricalFinancialDataFetcher(hf_token=hf_token)
    
    # Fetch a sample
    sample_df = historical.fetch_historical_data(
        start_year=2018,
        end_year=2020,
        max_articles_per_year=5000
    )
    
    print(f"\nSample: {len(sample_df)} articles")
    print(sample_df.head())


