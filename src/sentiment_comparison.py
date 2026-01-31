"""
Sentiment Feature Comparison Module
Compares raw daily sentiment scores vs rolling mean sentiment scores
Ensures no lookahead bias in all comparisons
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentFeatureComparator:
    """
    Compare different sentiment feature engineering approaches:
    - Raw daily sentiment scores
    - Rolling mean sentiment scores (various windows)
    """
    
    def __init__(self, windows: List[int] = [3, 7, 14, 30]):
        """
        Args:
            windows: List of rolling window sizes to test
        """
        self.windows = windows
        self.comparison_results = {}
    
    def create_sentiment_features(self, 
                                  sentiment_df: pd.DataFrame,
                                  sentiment_columns: List[str] = ['textblob', 'vader', 'finbert']
                                  ) -> pd.DataFrame:
        """
        Create both raw and rolling mean sentiment features
        
        Args:
            sentiment_df: DataFrame with date and sentiment scores
            sentiment_columns: List of sentiment method columns
            
        Returns:
            DataFrame with raw and rolling mean features
        """
        df = sentiment_df.copy()
        
        # Ensure date column
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        feature_df = df[['Date']].copy()
        
        # Add raw sentiment features
        for col in sentiment_columns:
            if col in df.columns:
                feature_df[f'{col}_raw'] = df[col]
                logger.info(f"Added raw feature: {col}_raw")
        
        # Add rolling mean features for each window
        for col in sentiment_columns:
            if col in df.columns:
                for window in self.windows:
                    # Use min_periods=1 to avoid NaN at the beginning
                    # This ensures we don't lose data points
                    feature_df[f'{col}_RM{window}'] = df[col].rolling(
                        window=window, 
                        min_periods=1
                    ).mean()
                    logger.info(f"Added rolling mean feature: {col}_RM{window}")
        
        return feature_df
    
    def compare_features_on_predictions(self,
                                       merged_data: pd.DataFrame,
                                       target_col: str = 'Close',
                                       train_ratio: float = 0.66,
                                       model_order: Tuple = (5, 1, 0)
                                       ) -> Dict[str, Dict[str, float]]:
        """
        Compare raw vs rolling mean sentiment features using SARIMAX predictions
        
        Args:
            merged_data: DataFrame with price and sentiment features
            target_col: Target column name (default: 'Close')
            train_ratio: Train/test split ratio
            model_order: SARIMAX order
            
        Returns:
            Dictionary with comparison results
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        results = {}
        
        # Get sentiment columns (raw and rolling means)
        sentiment_cols = [col for col in merged_data.columns 
                         if any(x in col for x in ['textblob', 'vader', 'finbert'])
                         and col != 'Date']
        
        target = merged_data[target_col].values
        size = int(len(target) * train_ratio)
        train, test = target[0:size], target[size:]
        
        logger.info(f"Train size: {len(train)}, Test size: {len(test)}")
        logger.info(f"Testing {len(sentiment_cols)} sentiment features")
        
        # Test each sentiment feature individually
        for sentiment_col in sentiment_cols:
            try:
                logger.info(f"\nTesting feature: {sentiment_col}")
                
                exog_all = merged_data[[sentiment_col]].values
                exog_train = exog_all[0:size]
                exog_test = exog_all[size:]
                
                # Walk-forward validation
                history = [x for x in train]
                history_exog = [x for x in exog_train]
                predictions = []
                
                for t in range(len(test)):
                    try:
                        model = SARIMAX(
                            history,
                            exog=np.array(history_exog).reshape(len(history_exog), -1),
                            order=model_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        model_fit = model.fit(disp=False, maxiter=100)
                        exog_next = exog_test[t].reshape(1, -1)
                        yhat = model_fit.forecast(steps=1, exog=exog_next)[0]
                    except Exception as e:
                        logger.debug(f"Forecast failed at step {t}: {e}")
                        yhat = history[-1]
                    
                    predictions.append(yhat)
                    obs = test[t]
                    history.append(obs)
                    history_exog.append(exog_test[t])
                    
                    if (t+1) % 20 == 0:
                        logger.debug(f'  Step {t+1}/{len(test)}')
                
                # Calculate metrics
                mae = mean_absolute_error(test, predictions)
                rmse = sqrt(mean_squared_error(test, predictions))
                mape = np.mean(np.abs((test - predictions) / test)) * 100
                
                results[sentiment_col] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'predictions': predictions
                }
                
                logger.info(f"  MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, MAPE: {mape:.2f}%")
                
            except Exception as e:
                logger.error(f"Error testing {sentiment_col}: {e}")
                continue
        
        self.comparison_results = results
        return results
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """
        Get summary DataFrame comparing all features
        
        Returns:
            DataFrame with comparison metrics
        """
        if not self.comparison_results:
            logger.warning("No comparison results available. Run compare_features_on_predictions first.")
            return pd.DataFrame()
        
        summary_data = []
        
        for feature_name, metrics in self.comparison_results.items():
            # Parse feature type
            if '_raw' in feature_name:
                feature_type = 'Raw'
                method = feature_name.replace('_raw', '')
            elif '_RM' in feature_name:
                parts = feature_name.split('_RM')
                method = parts[0]
                window = parts[1]
                feature_type = f'RM{window}'
            else:
                feature_type = 'Unknown'
                method = feature_name
            
            summary_data.append({
                'Feature': feature_name,
                'Method': method,
                'Type': feature_type,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE']
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('RMSE')
        
        return df
    
    def compare_raw_vs_rolling(self, method: str = 'textblob') -> Dict:
        """
        Direct comparison: raw vs best rolling mean for a specific method
        
        Args:
            method: Sentiment method to compare (textblob, vader, finbert)
            
        Returns:
            Dictionary with comparison results
        """
        if not self.comparison_results:
            logger.warning("No comparison results available.")
            return {}
        
        raw_key = f'{method}_raw'
        rm_keys = [k for k in self.comparison_results.keys() if k.startswith(method) and '_RM' in k]
        
        if raw_key not in self.comparison_results:
            logger.warning(f"Raw sentiment for {method} not found")
            return {}
        
        raw_metrics = self.comparison_results[raw_key]
        
        # Find best rolling mean
        best_rm_key = None
        best_rmse = float('inf')
        for rm_key in rm_keys:
            rmse = self.comparison_results[rm_key]['RMSE']
            if rmse < best_rmse:
                best_rmse = rmse
                best_rm_key = rm_key
        
        if best_rm_key is None:
            logger.warning(f"No rolling mean features found for {method}")
            return {
                'method': method,
                'raw': raw_metrics,
                'best_rolling_mean': None
            }
        
        rm_metrics = self.comparison_results[best_rm_key]
        
        # Calculate improvement
        improvement_rmse = ((raw_metrics['RMSE'] - rm_metrics['RMSE']) / raw_metrics['RMSE']) * 100
        improvement_mae = ((raw_metrics['MAE'] - rm_metrics['MAE']) / raw_metrics['MAE']) * 100
        improvement_mape = ((raw_metrics['MAPE'] - rm_metrics['MAPE']) / raw_metrics['MAPE']) * 100
        
        return {
            'method': method,
            'raw': {
                'feature': raw_key,
                'MAE': raw_metrics['MAE'],
                'RMSE': raw_metrics['RMSE'],
                'MAPE': raw_metrics['MAPE']
            },
            'best_rolling_mean': {
                'feature': best_rm_key,
                'MAE': rm_metrics['MAE'],
                'RMSE': rm_metrics['RMSE'],
                'MAPE': rm_metrics['MAPE']
            },
            'improvement': {
                'RMSE': improvement_rmse,
                'MAE': improvement_mae,
                'MAPE': improvement_mape
            }
        }
    
    def print_comparison_report(self, methods: List[str] = ['textblob', 'vader', 'finbert']):
        """
        Print detailed comparison report
        
        Args:
            methods: List of sentiment methods to compare
        """
        print("\n" + "="*80)
        print("SENTIMENT FEATURE COMPARISON REPORT")
        print("Raw Daily Sentiment vs Rolling Mean Sentiment")
        print("="*80)
        
        for method in methods:
            comparison = self.compare_raw_vs_rolling(method)
            
            if 'raw' not in comparison or 'best_rolling_mean' not in comparison:
                continue
            
            print(f"\n{'='*80}")
            print(f"Method: {method.upper()}")
            print(f"{'='*80}")
            
            raw = comparison['raw']
            rm = comparison['best_rolling_mean']
            imp = comparison['improvement']
            
            print(f"\nRaw Daily Sentiment ({raw['feature']}):")
            print(f"  MAE:  ${raw['MAE']:.2f}")
            print(f"  RMSE: ${raw['RMSE']:.2f}")
            print(f"  MAPE: {raw['MAPE']:.2f}%")
            
            if rm:
                print(f"\nBest Rolling Mean ({rm['feature']}):")
                print(f"  MAE:  ${rm['MAE']:.2f}")
                print(f"  RMSE: ${rm['RMSE']:.2f}")
                print(f"  MAPE: {rm['MAPE']:.2f}%")
                
                print(f"\nImprovement (Rolling Mean over Raw):")
                print(f"  RMSE: {imp['RMSE']:+.2f}%")
                print(f"  MAE:  {imp['MAE']:+.2f}%")
                print(f"  MAPE: {imp['MAPE']:+.2f}%")
                
                if imp['RMSE'] > 0:
                    print(f"\n✅ Rolling mean IMPROVES predictions by {imp['RMSE']:.2f}%")
                else:
                    print(f"\n❌ Raw sentiment performs better (by {-imp['RMSE']:.2f}%)")
        
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        summary_df = self.get_comparison_summary()
        print(summary_df.to_string(index=False))
        print("="*80 + "\n")
    
    def save_comparison_results(self, filepath: str):
        """
        Save comparison results to CSV
        
        Args:
            filepath: Output file path
        """
        summary_df = self.get_comparison_summary()
        summary_df.to_csv(filepath, index=False)
        logger.info(f"Comparison results saved to {filepath}")

