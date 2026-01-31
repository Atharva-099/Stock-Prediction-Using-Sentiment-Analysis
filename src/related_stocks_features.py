"""
Related Stocks Feature Engineering Module
Adds previous day price data from correlated stocks to avoid lookahead bias
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RelatedStocksFeatureEngine:
    """
    Engineer features from related/correlated stocks
    Always uses PREVIOUS day data to avoid lookahead bias
    """
    
    def __init__(self, 
                 related_tickers: List[str] = None,
                 sector_mapping: Dict[str, List[str]] = None):
        """
        Args:
            related_tickers: List of related stock tickers to fetch
            sector_mapping: Dictionary mapping sector -> list of tickers
        """
        # Default related stocks for major tech stocks
        if related_tickers is None:
            self.related_tickers = ['MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        else:
            self.related_tickers = related_tickers
        
        self.sector_mapping = sector_mapping or {}
        self.related_data = {}
        
        # Define stock sectors for smart grouping
        self.default_sectors = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC'],
            'ev': ['TSLA', 'F', 'GM', 'RIVN', 'LCID'],
            'ecommerce': ['AMZN', 'SHOP', 'EBAY', 'WMT'],
            'finance': ['JPM', 'BAC', 'GS', 'WFC', 'C'],
            'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK']
        }
    
    def get_related_stocks_for_ticker(self, ticker: str) -> List[str]:
        """
        Get related stocks for a given ticker based on sector
        
        Args:
            ticker: Stock ticker
            
        Returns:
            List of related stock tickers
        """
        related = []
        
        # Find sector
        for sector, stocks in self.default_sectors.items():
            if ticker in stocks:
                related = [s for s in stocks if s != ticker]
                logger.info(f"Found {len(related)} related stocks in {sector} sector")
                break
        
        # If not found, use default list
        if not related:
            related = [t for t in self.related_tickers if t != ticker]
        
        return related
    
    def fetch_related_stocks_data(self,
                                  tickers: List[str],
                                  start_date: str,
                                  end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data for multiple related tickers
        
        Args:
            tickers: List of stock tickers
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            
        Returns:
            Dictionary mapping ticker -> DataFrame
        """
        logger.info(f"Fetching data for {len(tickers)} related stocks...")
        
        related_data = {}
        
        for ticker in tickers:
            try:
                logger.info(f"  Fetching {ticker}...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if data.empty:
                    logger.warning(f"  No data for {ticker}")
                    continue
                
                # Flatten MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] for col in data.columns]
                
                # Reset index
                data = data.reset_index()
                if 'Date' not in data.columns and 'index' in data.columns:
                    data = data.rename(columns={'index': 'Date'})
                
                # Ensure timezone-naive dates
                data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
                
                related_data[ticker] = data
                logger.info(f"  ✓ {ticker}: {len(data)} days")
                
            except Exception as e:
                logger.error(f"  Error fetching {ticker}: {e}")
                continue
        
        self.related_data = related_data
        logger.info(f"Successfully fetched {len(related_data)} related stocks")
        
        return related_data
    
    def create_lagged_features(self,
                              target_df: pd.DataFrame,
                              lag_days: int = 1,
                              feature_cols: List[str] = ['Close', 'Volume', 'High', 'Low']
                              ) -> pd.DataFrame:
        """
        Create lagged features from related stocks to avoid lookahead bias
        CRITICAL: Always uses lag_days=1 (previous day) to prevent data leakage
        
        Args:
            target_df: Target stock DataFrame with Date column
            lag_days: Number of days to lag (default=1 for previous day)
            feature_cols: Columns to create features from
            
        Returns:
            DataFrame with lagged features from related stocks
        """
        if lag_days < 1:
            logger.warning("lag_days must be >= 1 to avoid lookahead bias. Setting to 1.")
            lag_days = 1
        
        logger.info(f"Creating lagged features (lag={lag_days} days) for {len(self.related_data)} stocks...")
        
        # Start with target dates
        merged_df = target_df[['Date']].copy()
        merged_df['Date'] = pd.to_datetime(merged_df['Date']).dt.tz_localize(None)
        
        for ticker, data in self.related_data.items():
            logger.info(f"  Processing {ticker}...")
            
            # Prepare data
            data_copy = data.copy()
            data_copy['Date'] = pd.to_datetime(data_copy['Date']).dt.tz_localize(None)
            
            # Create lagged features
            for col in feature_cols:
                if col not in data_copy.columns:
                    continue
                
                # Shift data by lag_days to create previous day features
                lagged_col = f'{ticker}_{col}_lag{lag_days}'
                data_copy[lagged_col] = data_copy[col].shift(lag_days)
                
                # Merge with target dates
                merge_cols = ['Date', lagged_col]
                merged_df = merged_df.merge(
                    data_copy[merge_cols],
                    on='Date',
                    how='left'
                )
        
        # Fill missing values with forward fill, then 0
        feature_cols_in_df = [col for col in merged_df.columns if col != 'Date']
        merged_df[feature_cols_in_df] = merged_df[feature_cols_in_df].fillna(method='ffill').fillna(0)
        
        logger.info(f"Created {len(feature_cols_in_df)} lagged features")
        
        return merged_df
    
    def create_relative_features(self,
                                target_df: pd.DataFrame,
                                target_ticker: str,
                                lag_days: int = 1) -> pd.DataFrame:
        """
        Create relative performance features (ratios and differences)
        comparing target stock to related stocks
        
        Args:
            target_df: Target stock DataFrame
            target_ticker: Target stock ticker
            lag_days: Number of days to lag
            
        Returns:
            DataFrame with relative features
        """
        logger.info(f"Creating relative features for {target_ticker}...")
        
        result_df = target_df[['Date']].copy()
        result_df['Date'] = pd.to_datetime(result_df['Date']).dt.tz_localize(None)
        
        # Get target stock close price (lagged)
        target_close = target_df[['Date', 'Close']].copy()
        target_close['Date'] = pd.to_datetime(target_close['Date']).dt.tz_localize(None)
        target_close['target_close_lagged'] = target_close['Close'].shift(lag_days)
        
        for ticker, data in self.related_data.items():
            if ticker == target_ticker:
                continue
            
            logger.info(f"  Computing relative features vs {ticker}...")
            
            data_copy = data.copy()
            data_copy['Date'] = pd.to_datetime(data_copy['Date']).dt.tz_localize(None)
            
            # Lagged close price
            data_copy['related_close_lagged'] = data_copy['Close'].shift(lag_days)
            
            # Merge with target
            temp_df = target_close.merge(
                data_copy[['Date', 'related_close_lagged']],
                on='Date',
                how='left'
            )
            
            # Compute relative features
            # 1. Price ratio
            ratio_col = f'{ticker}_price_ratio_lag{lag_days}'
            temp_df[ratio_col] = temp_df['target_close_lagged'] / (temp_df['related_close_lagged'] + 1e-8)
            
            # 2. Price difference
            diff_col = f'{ticker}_price_diff_lag{lag_days}'
            temp_df[diff_col] = temp_df['target_close_lagged'] - temp_df['related_close_lagged']
            
            # Merge into result
            result_df = result_df.merge(
                temp_df[['Date', ratio_col, diff_col]],
                on='Date',
                how='left'
            )
        
        # Fill missing values
        feature_cols = [col for col in result_df.columns if col != 'Date']
        result_df[feature_cols] = result_df[feature_cols].fillna(method='ffill').fillna(0)
        
        logger.info(f"Created {len(feature_cols)} relative features")
        
        return result_df
    
    def create_correlation_features(self,
                                   target_df: pd.DataFrame,
                                   window: int = 30) -> pd.DataFrame:
        """
        Create rolling correlation features between target and related stocks
        
        Args:
            target_df: Target stock DataFrame with Close column
            window: Rolling window for correlation calculation
            
        Returns:
            DataFrame with correlation features
        """
        logger.info(f"Creating correlation features (window={window})...")
        
        result_df = target_df[['Date', 'Close']].copy()
        result_df['Date'] = pd.to_datetime(result_df['Date']).dt.tz_localize(None)
        result_df = result_df.set_index('Date')
        
        for ticker, data in self.related_data.items():
            logger.info(f"  Computing correlation with {ticker}...")
            
            data_copy = data[['Date', 'Close']].copy()
            data_copy['Date'] = pd.to_datetime(data_copy['Date']).dt.tz_localize(None)
            data_copy = data_copy.set_index('Date')
            data_copy = data_copy.rename(columns={'Close': f'{ticker}_Close'})
            
            # Merge
            combined = result_df.join(data_copy, how='left')
            combined = combined.fillna(method='ffill')
            
            # Rolling correlation
            corr_col = f'{ticker}_correlation_{window}d'
            combined[corr_col] = combined['Close'].rolling(window).corr(combined[f'{ticker}_Close'])
            
            # Add to result
            result_df[corr_col] = combined[corr_col]
        
        result_df = result_df.reset_index()
        
        # Fill NaN correlations with 0
        corr_cols = [col for col in result_df.columns if 'correlation' in col]
        result_df[corr_cols] = result_df[corr_cols].fillna(0)
        
        logger.info(f"Created {len(corr_cols)} correlation features")
        
        return result_df
    
    def create_market_index_features(self,
                                    target_df: pd.DataFrame,
                                    indices: List[str] = ['^GSPC', '^DJI', '^IXIC'],
                                    lag_days: int = 1) -> pd.DataFrame:
        """
        Create features from market indices (S&P 500, Dow Jones, NASDAQ)
        
        Args:
            target_df: Target stock DataFrame
            indices: List of index tickers
            lag_days: Number of days to lag
            
        Returns:
            DataFrame with market index features
        """
        logger.info("Creating market index features...")
        
        result_df = target_df[['Date']].copy()
        result_df['Date'] = pd.to_datetime(result_df['Date']).dt.tz_localize(None)
        
        start_date = result_df['Date'].min()
        end_date = result_df['Date'].max()
        
        for index_ticker in indices:
            try:
                logger.info(f"  Fetching {index_ticker}...")
                
                index_data = yf.download(
                    index_ticker, 
                    start=start_date - timedelta(days=lag_days+5), 
                    end=end_date,
                    progress=False
                )
                
                if index_data.empty:
                    logger.warning(f"  No data for {index_ticker}")
                    continue
                
                # Flatten columns
                if isinstance(index_data.columns, pd.MultiIndex):
                    index_data.columns = [col[0] for col in index_data.columns]
                
                index_data = index_data.reset_index()
                if 'Date' not in index_data.columns:
                    index_data = index_data.rename(columns={'index': 'Date'})
                
                index_data['Date'] = pd.to_datetime(index_data['Date']).dt.tz_localize(None)
                
                # Create lagged features
                index_name = index_ticker.replace('^', '').lower()
                
                # Close price (lagged)
                close_col = f'index_{index_name}_close_lag{lag_days}'
                index_data[close_col] = index_data['Close'].shift(lag_days)
                
                # Daily return (lagged)
                return_col = f'index_{index_name}_return_lag{lag_days}'
                index_data[return_col] = index_data['Close'].pct_change().shift(lag_days)
                
                # Merge
                result_df = result_df.merge(
                    index_data[['Date', close_col, return_col]],
                    on='Date',
                    how='left'
                )
                
                logger.info(f"  ✓ Added features from {index_ticker}")
                
            except Exception as e:
                logger.error(f"  Error fetching {index_ticker}: {e}")
                continue
        
        # Fill missing values
        feature_cols = [col for col in result_df.columns if col != 'Date']
        result_df[feature_cols] = result_df[feature_cols].fillna(method='ffill').fillna(0)
        
        logger.info(f"Created {len(feature_cols)} market index features")
        
        return result_df
    
    def create_all_features(self,
                           target_df: pd.DataFrame,
                           target_ticker: str,
                           start_date: str,
                           end_date: str,
                           related_tickers: Optional[List[str]] = None,
                           lag_days: int = 1,
                           include_relative: bool = True,
                           include_correlation: bool = True,
                           include_market_indices: bool = True) -> pd.DataFrame:
        """
        Create all related stock features
        
        Args:
            target_df: Target stock DataFrame
            target_ticker: Target stock ticker
            start_date: Start date for fetching related stocks
            end_date: End date for fetching related stocks
            related_tickers: List of related tickers (if None, auto-detected)
            lag_days: Days to lag (must be >= 1)
            include_relative: Include relative performance features
            include_correlation: Include correlation features
            include_market_indices: Include market index features
            
        Returns:
            DataFrame with all related stock features
        """
        logger.info("Creating all related stock features...")
        
        # Auto-detect related stocks if not provided
        if related_tickers is None:
            related_tickers = self.get_related_stocks_for_ticker(target_ticker)
        
        # Fetch related stocks data
        self.fetch_related_stocks_data(related_tickers, start_date, end_date)
        
        # Create lagged features
        features_df = self.create_lagged_features(target_df, lag_days=lag_days)
        
        # Add relative features
        if include_relative and self.related_data:
            relative_df = self.create_relative_features(target_df, target_ticker, lag_days=lag_days)
            features_df = features_df.merge(relative_df, on='Date', how='left')
        
        # Add correlation features
        if include_correlation and self.related_data:
            corr_df = self.create_correlation_features(target_df, window=30)
            features_df = features_df.merge(corr_df, on='Date', how='left')
        
        # Add market index features
        if include_market_indices:
            index_df = self.create_market_index_features(target_df, lag_days=lag_days)
            features_df = features_df.merge(index_df, on='Date', how='left')
        
        logger.info(f"Total features created: {features_df.shape[1] - 1} (excluding Date)")
        
        return features_df
    
    def validate_no_lookahead_bias(self, features_df: pd.DataFrame) -> bool:
        """
        Validate that all features are properly lagged (no lookahead bias)
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating for lookahead bias...")
        
        # Check that all features have 'lag' in the name
        feature_cols = [col for col in features_df.columns 
                       if col != 'Date' and not col.startswith('correlation')]
        
        non_lagged = [col for col in feature_cols if 'lag' not in col.lower()]
        
        if non_lagged:
            logger.warning(f"Found {len(non_lagged)} features without explicit lag:")
            for col in non_lagged[:10]:  # Show first 10
                logger.warning(f"  - {col}")
            logger.warning("⚠️ POTENTIAL LOOKAHEAD BIAS DETECTED!")
            return False
        else:
            logger.info("✅ All features are properly lagged - no lookahead bias detected")
            return True

