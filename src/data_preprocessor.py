"""
Data Preprocessing Module
Implements log returns, stationarity testing, and feature engineering
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, List, Optional, Dict
from statsmodels.tsa.stattools import adfuller, kpss
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataProcessor:
    """
    Process stock market data with focus on stationarity
    """
    
    def __init__(self, use_log_returns: bool = True):
        """
        Args:
            use_log_returns: Use log returns instead of raw prices for stationarity
        """
        self.use_log_returns = use_log_returns
    
    @staticmethod
    def fetch_stock_data(ticker: str, 
                        start_date: str, 
                        end_date: str) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            
        Returns:
            DataFrame with stock data
        """
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                logger.error(f"No data fetched for {ticker}")
                return pd.DataFrame()
            
            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            # Reset index and ensure Date column
            data = data.reset_index()
            if 'Date' not in data.columns and 'index' in data.columns:
                data = data.rename(columns={'index': 'Date'})
            
            # Ensure timezone-naive dates
            data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
            
            logger.info(f"Fetched {len(data)} rows for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def compute_log_returns(prices: pd.Series) -> pd.Series:
        """
        Compute log returns: log(P_t / P_{t-1})
        Log returns are more suitable for time series modeling as they:
        - Are additive over time
        - Better approximate normal distribution
        - Help achieve stationarity
        
        Args:
            prices: Price series
            
        Returns:
            Log returns series
        """
        log_returns = np.log(prices / prices.shift(1))
        return log_returns.dropna()
    
    @staticmethod
    def compute_simple_returns(prices: pd.Series) -> pd.Series:
        """
        Compute simple returns: (P_t - P_{t-1}) / P_{t-1}
        
        Args:
            prices: Price series
            
        Returns:
            Simple returns series
        """
        returns = prices.pct_change()
        return returns.dropna()
    
    def test_stationarity(self, series: pd.Series, 
                         method: str = 'adf',
                         significance_level: float = 0.05) -> Tuple[bool, float, Dict]:
        """
        Test time series for stationarity
        
        Args:
            series: Time series to test
            method: 'adf' for Augmented Dickey-Fuller or 'kpss' for KPSS test
            significance_level: P-value threshold
            
        Returns:
            (is_stationary, p_value, test_results_dict)
        """
        series = series.dropna()
        
        if len(series) < 10:
            logger.warning("Series too short for stationarity test")
            return False, 1.0, {}
        
        try:
            if method.lower() == 'adf':
                # ADF test: H0 = series has unit root (non-stationary)
                # Reject H0 if p-value < significance_level => stationary
                result = adfuller(series, autolag='AIC')
                p_value = result[1]
                is_stationary = p_value < significance_level
                
                test_results = {
                    'test_statistic': result[0],
                    'p_value': p_value,
                    'n_lags': result[2],
                    'n_obs': result[3],
                    'critical_values': result[4],
                    'method': 'ADF'
                }
                
                logger.info(f"ADF Test - Statistic: {result[0]:.4f}, p-value: {p_value:.4f}, Stationary: {is_stationary}")
                
            elif method.lower() == 'kpss':
                # KPSS test: H0 = series is stationary
                # Reject H0 if p-value < significance_level => non-stationary
                result = kpss(series, regression='c', nlags='auto')
                p_value = result[1]
                is_stationary = p_value >= significance_level  # Note: opposite of ADF
                
                test_results = {
                    'test_statistic': result[0],
                    'p_value': p_value,
                    'n_lags': result[2],
                    'critical_values': result[3],
                    'method': 'KPSS'
                }
                
                logger.info(f"KPSS Test - Statistic: {result[0]:.4f}, p-value: {p_value:.4f}, Stationary: {is_stationary}")
            
            else:
                raise ValueError(f"Unknown method: {method}. Use 'adf' or 'kpss'")
            
            return is_stationary, p_value, test_results
            
        except Exception as e:
            logger.error(f"Stationarity test failed: {e}")
            return False, 1.0, {}
    
    def find_differencing_order(self, series: pd.Series, 
                               max_order: int = 3,
                               method: str = 'adf') -> int:
        """
        Find optimal differencing order to achieve stationarity
        
        Args:
            series: Time series
            max_order: Maximum differencing order to try
            method: Stationarity test method
            
        Returns:
            Optimal differencing order (0 if already stationary)
        """
        # Test original series
        is_stationary, p_value, _ = self.test_stationarity(series, method=method)
        
        if is_stationary:
            logger.info("Series is already stationary (d=0)")
            return 0
        
        # Try differencing orders
        for d in range(1, max_order + 1):
            diff_series = series.diff(periods=d).dropna()
            is_stationary, p_value, _ = self.test_stationarity(diff_series, method=method)
            
            if is_stationary:
                logger.info(f"Stationarity achieved with differencing order d={d}")
                return d
        
        logger.warning(f"Stationarity not achieved up to d={max_order}. Using d={max_order}")
        return max_order
    
    def add_rolling_features(self, df: pd.DataFrame, 
                            columns: List[str],
                            windows: List[int] = [7, 14]) -> pd.DataFrame:
        """
        Add rolling mean features (as per research paper)
        
        Args:
            df: Input DataFrame
            columns: Columns to compute rolling means for
            windows: Rolling window sizes (default: 7 and 14 days)
            
        Returns:
            DataFrame with added rolling mean columns
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
            
            for window in windows:
                new_col = f"{col}_RM{window}"
                df[new_col] = df[col].rolling(window=window, min_periods=1).mean()
                logger.info(f"Added rolling mean feature: {new_col}")
        
        return df
    
    def prepare_time_series_data(self, df: pd.DataFrame,
                                target_col: str = 'Close',
                                date_col: str = 'Date') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for time series modeling
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            date_col: Date column name
            
        Returns:
            (processed_df, target_series)
        """
        df = df.copy()
        
        # Ensure date is datetime
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Extract target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        target = df[target_col].copy()
        
        # Apply log returns if enabled
        if self.use_log_returns and target_col in ['Close', 'Open', 'High', 'Low']:
            logger.info(f"Computing log returns for {target_col}")
            original_target = target.copy()
            target = self.compute_log_returns(target)
            
            # Update DataFrame with log returns
            df['original_' + target_col] = original_target
            df[target_col + '_log_return'] = target
            
            # Realign indices after differencing
            df = df.iloc[1:].reset_index(drop=True)
            target = target.reset_index(drop=True)
        
        logger.info(f"Prepared data: {len(df)} observations")
        return df, target
    
    def merge_with_sentiment(self, price_df: pd.DataFrame,
                            sentiment_df: pd.DataFrame,
                            date_col: str = 'Date') -> pd.DataFrame:
        """
        Merge price data with sentiment data
        
        Args:
            price_df: Price DataFrame with Date column
            sentiment_df: Sentiment DataFrame with date column
            date_col: Date column name in price_df
            
        Returns:
            Merged DataFrame
        """
        price_df = price_df.copy()
        sentiment_df = sentiment_df.copy()
        
        # Ensure both have date columns
        price_df[date_col] = pd.to_datetime(price_df[date_col]).dt.date
        
        if 'date' in sentiment_df.columns:
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
        elif 'Date' in sentiment_df.columns:
            sentiment_df = sentiment_df.rename(columns={'Date': 'date'})
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
        
        # Merge
        merged = price_df.merge(
            sentiment_df, 
            left_on=date_col, 
            right_on='date',
            how='left'
        )
        
        # Fill missing sentiment with 0 (neutral)
        sentiment_cols = [col for col in merged.columns if 'sentiment' in col.lower() or 'score' in col.lower()]
        for col in sentiment_cols:
            merged[col] = merged[col].fillna(0.0)
        
        # Drop duplicate date column
        if 'date' in merged.columns and date_col in merged.columns and date_col != 'date':
            merged = merged.drop(columns=['date'])
        
        logger.info(f"Merged data: {len(merged)} rows with {len(sentiment_cols)} sentiment features")
        return merged
    
    def create_train_test_split(self, df: pd.DataFrame,
                               train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets (time-aware)
        
        Args:
            df: Input DataFrame
            train_ratio: Proportion of data for training
            
        Returns:
            (train_df, test_df)
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        return train_df, test_df


def create_feature_combinations(df: pd.DataFrame,
                               sentiment_methods: List[str] = ['textblob', 'vader', 'finbert'],
                               use_rolling_means: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Create different feature combinations for experimentation
    Based on research paper's 67 feature setups
    
    Args:
        df: Input DataFrame with price and sentiment features
        sentiment_methods: List of sentiment methods to use
        use_rolling_means: Whether to include rolling mean features
        
    Returns:
        Dictionary of feature setup name -> DataFrame
    """
    feature_sets = {}
    
    # 1. Baseline: Price only
    price_cols = ['Close']
    if all(col in df.columns for col in price_cols):
        feature_sets['price_only'] = df[['Date'] + price_cols].copy()
    
    # 2. Price + individual sentiment
    for method in sentiment_methods:
        sentiment_col = f'{method}_sentiment'
        if sentiment_col in df.columns:
            cols = ['Date', 'Close', sentiment_col]
            feature_sets[f'price_{method}'] = df[cols].copy()
    
    # 3. Price + rolling means
    if use_rolling_means:
        rm_cols = ['Date', 'Close']
        for window in [7, 14]:
            col = f'Close_RM{window}'
            if col in df.columns:
                rm_cols.append(col)
        
        if len(rm_cols) > 2:
            feature_sets['price_rolling_means'] = df[rm_cols].copy()
    
    # 4. Price + rolling means + sentiment (optimal per research)
    if use_rolling_means:
        for method in sentiment_methods:
            sentiment_col = f'{method}_sentiment'
            sentiment_rm7 = f'{method}_sentiment_RM7'
            
            if sentiment_col in df.columns:
                cols = ['Date', 'Close', 'Close_RM7', sentiment_col]
                
                if sentiment_rm7 in df.columns:
                    cols.append(sentiment_rm7)
                
                feature_sets[f'optimal_{method}'] = df[cols].copy()
    
    # 5. All features
    all_cols = ['Date', 'Close']
    for col in df.columns:
        if any(x in col for x in ['sentiment', 'RM']):
            all_cols.append(col)
    
    if len(all_cols) > 2:
        feature_sets['all_features'] = df[all_cols].copy()
    
    logger.info(f"Created {len(feature_sets)} feature combinations")
    return feature_sets

