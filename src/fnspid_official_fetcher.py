"""
Official FNSPID Dataset Fetcher
================================
Fetches from the official FNSPID dataset at Zihan1004/FNSPID on HuggingFace
This dataset covers 1999-2023 with 15.7M financial news articles

Reference: https://github.com/Zdong104/FNSPID_Financial_News_Dataset
Paper: https://arxiv.org/abs/2402.06698

Author: CMU Financial Forecasting Project
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import requests
from tqdm import tqdm
import zipfile
import io

logger = logging.getLogger(__name__)


class OfficialFNSPIDFetcher:
    """
    Fetches from the official FNSPID dataset on HuggingFace
    Dataset: Zihan1004/FNSPID
    
    Contains:
    - 29.7 million stock prices
    - 15.7 million financial news records
    - 4,775 S&P 500 companies
    - Date range: 1999-2023
    """
    
    def __init__(self, cache_dir='data/fnspid_official'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Official FNSPID dataset URLs from HuggingFace
        self.news_url = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"
        self.stock_url = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip"
    
    def download_news_data(self, force_download=False):
        """
        Download the FNSPID news data (nasdaq_external_data.csv)
        """
        news_file = os.path.join(self.cache_dir, 'nasdaq_external_data.csv')
        
        if os.path.exists(news_file) and not force_download:
            logger.info(f"Using cached news data: {news_file}")
            return news_file
        
        logger.info("Downloading FNSPID news data from HuggingFace...")
        logger.info(f"URL: {self.news_url}")
        
        try:
            response = requests.get(self.news_url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(news_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"✓ Downloaded to: {news_file}")
            return news_file
            
        except Exception as e:
            logger.error(f"Error downloading news data: {e}")
            return None
    
    def download_stock_data(self, force_download=False):
        """
        Download the FNSPID stock price data (full_history.zip)
        """
        zip_file = os.path.join(self.cache_dir, 'full_history.zip')
        extract_dir = os.path.join(self.cache_dir, 'stock_prices')
        
        if os.path.exists(extract_dir) and not force_download:
            logger.info(f"Using cached stock data: {extract_dir}")
            return extract_dir
        
        logger.info("Downloading FNSPID stock data from HuggingFace...")
        logger.info(f"URL: {self.stock_url}")
        
        try:
            response = requests.get(self.stock_url, stream=True, timeout=600)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Extract
            logger.info("Extracting stock data...")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info(f"✓ Extracted to: {extract_dir}")
            return extract_dir
            
        except Exception as e:
            logger.error(f"Error downloading stock data: {e}")
            return None
    
    def load_news_data(self, start_year=1999, end_year=2023, ticker=None):
        """
        Load news data for specified year range
        
        Args:
            start_year: Starting year (1999-2023)
            end_year: Ending year (1999-2023)
            ticker: Optional - filter by stock ticker
            
        Returns:
            DataFrame with news articles
        """
        news_file = self.download_news_data()
        
        if news_file is None:
            logger.error("Could not download news data")
            return pd.DataFrame()
        
        logger.info(f"Loading news data for {start_year}-{end_year}...")
        
        try:
            # Read CSV
            df = pd.read_csv(news_file)
            
            logger.info(f"Loaded {len(df):,} total articles")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Parse dates
            date_cols = ['date', 'Date', 'publish_date', 'published_date', 'timestamp']
            date_col = None
            for col in date_cols:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df['date'] = pd.to_datetime(df[date_col], errors='coerce')
                df['year'] = df['date'].dt.year
                
                # Filter by year range
                df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
                logger.info(f"Filtered to {len(df):,} articles ({start_year}-{end_year})")
            
            # Filter by ticker if specified
            if ticker:
                ticker_cols = ['ticker', 'Ticker', 'symbol', 'stock']
                for col in ticker_cols:
                    if col in df.columns:
                        df = df[df[col].str.upper() == ticker.upper()]
                        logger.info(f"Filtered to {len(df):,} articles for {ticker}")
                        break
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading news data: {e}")
            return pd.DataFrame()
    
    def fetch_news_for_years(self, years_needed, max_per_year=50000):
        """
        Fetch news data for specific years
        
        Args:
            years_needed: List of years to fetch (e.g., [1999, 2000, 2021, 2022, 2023])
            max_per_year: Maximum articles per year
            
        Returns:
            DataFrame with news articles
        """
        logger.info("=" * 80)
        logger.info("FETCHING FROM OFFICIAL FNSPID DATASET (Zihan1004/FNSPID)")
        logger.info("=" * 80)
        logger.info(f"Years needed: {years_needed}")
        
        all_news = self.load_news_data(
            start_year=min(years_needed),
            end_year=max(years_needed)
        )
        
        if all_news.empty:
            return pd.DataFrame()
        
        # Filter to only needed years
        all_news = all_news[all_news['year'].isin(years_needed)]
        
        # Sample if too many per year
        result_dfs = []
        for year in years_needed:
            year_data = all_news[all_news['year'] == year]
            if len(year_data) > max_per_year:
                year_data = year_data.sample(n=max_per_year, random_state=42)
            result_dfs.append(year_data)
            logger.info(f"  {year}: {len(year_data):,} articles")
        
        if result_dfs:
            result = pd.concat(result_dfs, ignore_index=True)
            logger.info(f"\n✓ Total: {len(result):,} articles for {len(years_needed)} years")
            return result
        
        return pd.DataFrame()
    
    def get_yearly_distribution(self):
        """Get article counts per year"""
        all_news = self.load_news_data(start_year=1999, end_year=2025)
        
        if all_news.empty:
            return {}
        
        return all_news.groupby('year').size().to_dict()


class CombinedDataFetcher:
    """
    Combines data from multiple sources to maximize coverage:
    1. Official FNSPID (Zihan1004/FNSPID) - 1999-2023
    2. Multi-source (Brianferrell787/financial-news-multisource) - 2009-2020
    3. ashraq/financial-news - 2010-2020
    4. RSS Feeds - 2024-2025
    """
    
    def __init__(self, hf_token=None, cache_dir='data'):
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        
        self.fnspid_official = OfficialFNSPIDFetcher(
            cache_dir=os.path.join(cache_dir, 'fnspid_official')
        )
    
    def fetch_all_available_data(self, start_year=1999, end_year=2025):
        """
        Fetch data from all available sources for maximum coverage
        """
        from src.historical_data_fetcher import HistoricalFinancialDataFetcher, RecentDataFetcher
        
        logger.info("=" * 100)
        logger.info(" COMBINED DATA FETCH: ALL SOURCES (1999-2025)")
        logger.info("=" * 100)
        
        all_data = []
        year_coverage = {}
        
        # 1. Official FNSPID for 1999-2008 and 2021-2023
        fnspid_years = list(range(1999, 2009)) + list(range(2021, 2024))
        logger.info(f"\n[Source 1] Official FNSPID (Zihan1004/FNSPID)")
        logger.info(f"  Target years: {fnspid_years}")
        
        fnspid_data = self.fnspid_official.fetch_news_for_years(fnspid_years)
        if not fnspid_data.empty:
            # Standardize columns
            fnspid_data = self._standardize_columns(fnspid_data, 'FNSPID_Official')
            all_data.append(fnspid_data)
            for year in fnspid_data['year'].unique():
                year_coverage[year] = year_coverage.get(year, []) + ['FNSPID_Official']
        
        # 2. Multi-source for 2009-2020 (already running in main script)
        logger.info(f"\n[Source 2] Multi-source dataset (Brianferrell787)")
        logger.info(f"  Years 2009-2020 being fetched by main pipeline")
        
        # 3. RSS Feeds for 2024-2025
        logger.info(f"\n[Source 3] RSS Feeds")
        recent_fetcher = RecentDataFetcher(hf_token=self.hf_token)
        rss_data = recent_fetcher.fetch_rss_news(query="stock market", days_back=365)
        if not rss_data.empty:
            rss_data = self._standardize_columns(rss_data, 'RSS')
            all_data.append(rss_data)
            for year in rss_data['year'].unique():
                year_coverage[year] = year_coverage.get(year, []) + ['RSS']
        
        # Combine all
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.drop_duplicates(subset=['date', 'text'], keep='first')
            combined = combined.sort_values('date')
            
            logger.info("\n" + "=" * 60)
            logger.info("COMBINED DATA SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total articles: {len(combined):,}")
            logger.info(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
            logger.info("\nYear coverage:")
            for year in sorted(year_coverage.keys()):
                sources = year_coverage[year]
                count = len(combined[combined['year'] == year])
                logger.info(f"  {year}: {count:,} articles from {', '.join(set(sources))}")
            
            return combined
        
        return pd.DataFrame()
    
    def _standardize_columns(self, df, source_name):
        """Standardize column names across datasets"""
        # Ensure required columns
        if 'date' not in df.columns:
            date_cols = ['Date', 'publish_date', 'published_date', 'timestamp']
            for col in date_cols:
                if col in df.columns:
                    df['date'] = pd.to_datetime(df[col], errors='coerce')
                    break
        
        if 'text' not in df.columns:
            text_cols = ['title', 'headline', 'content', 'article', 'full_text']
            for col in text_cols:
                if col in df.columns:
                    df['text'] = df[col]
                    break
        
        if 'year' not in df.columns:
            df['year'] = pd.to_datetime(df['date']).dt.year
        
        df['source'] = source_name
        
        return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test official FNSPID fetcher
    fetcher = OfficialFNSPIDFetcher()
    
    # Get years 1999-2008 and 2021-2023
    years_needed = list(range(1999, 2009)) + list(range(2021, 2024))
    news_df = fetcher.fetch_news_for_years(years_needed, max_per_year=10000)
    
    if not news_df.empty:
        print(f"\nFetched {len(news_df):,} articles")
        print(f"Columns: {news_df.columns.tolist()}")
        print(f"\nYearly distribution:")
        print(news_df['year'].value_counts().sort_index())


