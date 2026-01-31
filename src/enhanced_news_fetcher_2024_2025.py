"""
Enhanced News Fetcher for 2024-2025
====================================
Uses MULTIPLE sources to get more 2024-2025 data:
1. Multiple RSS feeds with different queries
2. Google News RSS
3. Yahoo Finance RSS
4. CNBC, MarketWatch, Bloomberg RSS

Author: CMU Financial Forecasting Project
Date: November 2025
"""

import feedparser
import pandas as pd
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


class EnhancedNewsFetcher2024_2025:
    """
    Fetches 2024-2025 news from multiple RSS sources
    Goal: Get thousands of articles instead of just 40
    """
    
    def __init__(self):
        # Multiple search queries for comprehensive coverage
        self.search_queries = [
            # General market
            "stock market",
            "S&P 500",
            "NASDAQ",
            "Dow Jones",
            "market news",
            "trading news",
            
            # Tech stocks (including AAPL)
            "Apple stock",
            "Apple Inc",
            "AAPL",
            "tech stocks",
            "technology sector",
            "iPhone sales",
            "Apple earnings",
            
            # Economic factors
            "Federal Reserve",
            "interest rates",
            "inflation",
            "unemployment",
            "GDP",
            "economic outlook",
            
            # Market events
            "earnings report",
            "quarterly results",
            "stock rally",
            "market crash",
            "stock split",
            "dividend",
            
            # Sector news
            "semiconductor",
            "chip stocks",
            "EV stocks",
            "AI stocks",
            "cloud computing",
            
            # Global events
            "China trade",
            "supply chain",
            "oil prices",
            "global markets",
        ]
        
        # RSS feed sources
        self.rss_sources = {
            'google_news': 'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en',
            'yahoo_finance': 'https://finance.yahoo.com/rss/topstories',
            'cnbc': 'https://www.cnbc.com/id/10001147/device/rss/rss.html',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'reuters': 'https://news.google.com/rss/search?q=site:reuters.com+{query}&hl=en-US&gl=US',
            'bloomberg': 'https://news.google.com/rss/search?q=site:bloomberg.com+{query}&hl=en-US&gl=US',
            'wsj': 'https://news.google.com/rss/search?q=site:wsj.com+{query}&hl=en-US&gl=US',
        }
    
    def fetch_rss_feed(self, url, source_name, max_articles=100):
        """Fetch articles from a single RSS feed"""
        articles = []
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_articles]:
                try:
                    # Parse date
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])
                    else:
                        pub_date = datetime.now()
                    
                    # Only keep 2024-2025 articles
                    if pub_date.year >= 2024:
                        articles.append({
                            'Date': pub_date,
                            'Article_title': entry.get('title', ''),
                            'text': entry.get('summary', entry.get('title', '')),
                            'source': source_name,
                            'link': entry.get('link', '')
                        })
                except Exception as e:
                    continue
        except Exception as e:
            logger.debug(f"Failed to fetch {source_name}: {e}")
        
        return articles
    
    def fetch_google_news_query(self, query):
        """Fetch from Google News for a specific query"""
        url = self.rss_sources['google_news'].format(query=query.replace(' ', '+'))
        return self.fetch_rss_feed(url, f'Google_{query}', max_articles=50)
    
    def fetch_all_news(self, days_back=365):
        """
        Fetch news from ALL sources for maximum coverage
        Returns thousands of articles for 2024-2025
        """
        logger.info("=" * 80)
        logger.info("ENHANCED NEWS FETCHER: 2024-2025")
        logger.info("Fetching from multiple sources...")
        logger.info("=" * 80)
        
        all_articles = []
        
        # 1. Fetch from all search queries via Google News
        logger.info(f"\n[1] Google News - {len(self.search_queries)} queries...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.fetch_google_news_query, q): q 
                      for q in self.search_queries}
            
            for future in as_completed(futures):
                query = futures[future]
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                    if articles:
                        logger.info(f"  ✓ '{query}': {len(articles)} articles")
                except Exception as e:
                    logger.debug(f"  ✗ '{query}' failed: {e}")
        
        logger.info(f"  Total from Google News: {len(all_articles)}")
        
        # 2. Fetch from static RSS feeds
        logger.info("\n[2] Financial news RSS feeds...")
        
        static_feeds = [
            ('Yahoo Finance', self.rss_sources['yahoo_finance']),
            ('CNBC', self.rss_sources['cnbc']),
            ('MarketWatch', self.rss_sources['marketwatch']),
        ]
        
        for name, url in static_feeds:
            articles = self.fetch_rss_feed(url, name, max_articles=200)
            all_articles.extend(articles)
            logger.info(f"  ✓ {name}: {len(articles)} articles")
        
        # 3. Fetch from major news sites via Google
        logger.info("\n[3] Major financial sites via Google...")
        
        major_queries = ['stock market', 'earnings', 'Apple']
        for source in ['reuters', 'bloomberg', 'wsj']:
            for query in major_queries:
                url = self.rss_sources[source].format(query=query.replace(' ', '+'))
                articles = self.fetch_rss_feed(url, f'{source}_{query}', max_articles=30)
                all_articles.extend(articles)
                if articles:
                    logger.info(f"  ✓ {source}/{query}: {len(articles)} articles")
                time.sleep(0.5)  # Rate limiting
        
        # Convert to DataFrame
        if all_articles:
            df = pd.DataFrame(all_articles)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['Article_title'], keep='first')
            
            # Filter to 2024-2025 only
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[df['Date'].dt.year >= 2024]
            
            # Sort by date
            df = df.sort_values('Date', ascending=False)
            
            logger.info("\n" + "=" * 60)
            logger.info("ENHANCED FETCHER RESULTS")
            logger.info("=" * 60)
            logger.info(f"Total unique articles: {len(df):,}")
            logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            logger.info(f"2024 articles: {len(df[df['Date'].dt.year == 2024]):,}")
            logger.info(f"2025 articles: {len(df[df['Date'].dt.year == 2025]):,}")
            
            return df
        
        return pd.DataFrame()
    
    def save_to_parquet(self, save_path='data/fnspid_official/news_2024_2025_enhanced.parquet'):
        """Fetch and save to parquet"""
        df = self.fetch_all_news()
        
        if not df.empty:
            df.to_parquet(save_path, index=False)
            logger.info(f"\n✓ Saved {len(df):,} articles to {save_path}")
        
        return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    fetcher = EnhancedNewsFetcher2024_2025()
    df = fetcher.save_to_parquet()
    
    if not df.empty:
        print(f"\n✅ Fetched {len(df):,} articles for 2024-2025")
        print(f"\nSample articles:")
        print(df[['Date', 'Article_title', 'source']].head(10))


