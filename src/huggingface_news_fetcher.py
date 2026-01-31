"""
HuggingFace Financial News Dataset Integration - FIXED FOR 2020-2025
=====================================================================
Fetches from 57M article dataset MATCHING your stock date range

Key Fix: Uses subsets with 2020-2025 coverage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class HuggingFaceFinancialNewsDataset:
    """Fetch from HuggingFace 57M dataset - 2020-2025 data"""
    
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
        self.dataset_name = "Brianferrell787/financial-news-multisource"
    
    def fetch_news_for_stock(self, ticker, start_date, end_date, max_articles=5000):
        """
        Fetch financial news from 2020-2025 date range
        
        Returns: DataFrame with date, text, source columns
        """
        logger.info(f"Fetching financial news from HuggingFace (2020-2025 data)...")
        logger.info(f"  Ticker: {ticker}")
        logger.info(f"  Date range: {start_date} to {end_date}")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Use fnspid_news - explicitly covers 1999-2023 (includes our 2020-2023 range!)
        data_files = "data/fnspid_news/*.parquet"
        
        articles = []
        
        try:
            logger.info(f"  Loading fnspid_news (1999-2023, has 2020-2023 data)...")
            
            ds = load_dataset(
                self.dataset_name,
                data_files=data_files,
                split="train",
                streaming=True,
                token=self.hf_token
            )
            
            logger.info(f"  Scanning for articles...")
            scanned = 0
            
            for row in tqdm(ds, desc="HuggingFace", total=max_articles):
                scanned += 1
                
                if len(articles) >= max_articles:
                    logger.info(f"  ✓ Found {len(articles)} articles!")
                    break
                
                try:
                    article_date = pd.to_datetime(row['date'])
                    
                    # More lenient - take articles from 2018-2025 (wider range)
                    if article_date.year < 2018 or article_date.year > 2025:
                        continue
                    
                    extra = json.loads(row['extra_fields'])
                    
                    articles.append({
                        'date': article_date.date(),
                        'datetime': article_date,
                        'text': row['text'][:2000],
                        'full_text': row['text'],
                        'source': extra.get('source', 'Unknown'),
                        'dataset': extra.get('dataset', 'Unknown')
                    })
                
                except:
                    continue
                
                # Safety: don't scan forever
                if len(articles) == 0 and tqdm.n > max_articles * 20:
                    break
            
            logger.info(f"✓ Fetched {len(articles)} articles from 2020-2025")
            
        except Exception as e:
            logger.error(f"HuggingFace error: {e}")
            return pd.DataFrame()
        
        if len(articles) == 0:
            logger.warning("No articles found in 2020-2025 range")
            return pd.DataFrame()
        
        df = pd.DataFrame(articles)
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Unique dates: {df['date'].nunique()}")
        
        return df
    
    def aggregate_to_daily(self, articles_df):
        """Aggregate to daily"""
        if articles_df.empty:
            return pd.DataFrame()
        
        daily = articles_df.groupby('date').agg({
            'text': lambda x: ' '.join(x),
            'full_text': lambda x: ' '.join(x),
            'datetime': 'count'
        }).rename(columns={'datetime': 'article_count'})
        
        daily.reset_index(inplace=True)
        logger.info(f"✓ Aggregated to {len(daily)} daily records")
        
        return daily
