"""
Comprehensive Evaluation Metrics Module
Implements all metrics from research paper:
- MAE, MAPE, MSE, RMSE, RMSLE, R²
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value (as percentage)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        return np.inf
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def root_mean_squared_logarithmic_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Logarithmic Error (RMSLE)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSLE value
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Ensure non-negative values for log
    y_true = np.abs(y_true)
    y_pred = np.abs(y_pred)
    
    rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
    return rmsle


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics as per research paper
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metric names and values
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        logger.warning("No valid data points for metric calculation")
        return {metric: np.nan for metric in ['mae', 'mape', 'mse', 'rmse', 'rmsle', 'r2']}
    
    metrics = {}
    
    try:
        # MAE: Mean Absolute Error
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # MAPE: Mean Absolute Percentage Error
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        
        # MSE: Mean Squared Error
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        
        # RMSE: Root Mean Squared Error
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # RMSLE: Root Mean Squared Logarithmic Error
        metrics['rmsle'] = root_mean_squared_logarithmic_error(y_true, y_pred)
        
        # R²: Coefficient of Determination
        metrics['r2'] = r2_score(y_true, y_pred)
        
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        metrics = {metric: np.nan for metric in ['mae', 'mape', 'mse', 'rmse', 'rmsle', 'r2']}
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Pretty print evaluation metrics
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Metrics for {model_name}")
    print(f"{'='*60}")
    
    metric_descriptions = {
        'mae': 'Mean Absolute Error (MAE)',
        'mape': 'Mean Absolute Percentage Error (MAPE)',
        'mse': 'Mean Squared Error (MSE)',
        'rmse': 'Root Mean Squared Error (RMSE)',
        'rmsle': 'Root Mean Squared Logarithmic Error (RMSLE)',
        'r2': 'R² Score (Coefficient of Determination)'
    }
    
    for metric, value in metrics.items():
        description = metric_descriptions.get(metric, metric.upper())
        if metric == 'mape':
            print(f"{description:50s}: {value:>10.2f}%")
        else:
            print(f"{description:50s}: {value:>10.6f}")
    
    print(f"{'='*60}\n")


def compare_models(results: Dict[str, Dict[str, float]], 
                  metric: str = 'rmse') -> pd.DataFrame:
    """
    Compare multiple models based on evaluation metrics
    
    Args:
        results: Dictionary of {model_name: {metric: value}}
        metric: Metric to sort by
        
    Returns:
        DataFrame with comparison results
    """
    df = pd.DataFrame(results).T
    
    # Sort by specified metric (ascending for error metrics, descending for R²)
    ascending = metric != 'r2'
    df = df.sort_values(by=metric, ascending=ascending)
    
    # Add ranking
    df['rank'] = range(1, len(df) + 1)
    
    return df


class MetricsTracker:
    """
    Track and store metrics across multiple experiments
    """
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, 
                  experiment_name: str,
                  y_true: np.ndarray,
                  y_pred: np.ndarray,
                  metadata: Dict = None):
        """
        Add experiment result
        
        Args:
            experiment_name: Name of the experiment
            y_true: True values
            y_pred: Predicted values
            metadata: Additional metadata (model params, etc.)
        """
        metrics = compute_all_metrics(y_true, y_pred)
        
        self.results[experiment_name] = {
            'metrics': metrics,
            'metadata': metadata or {},
            'n_samples': len(y_true)
        }
        
        logger.info(f"Added results for: {experiment_name}")
    
    def get_comparison_df(self, metric: str = 'rmse') -> pd.DataFrame:
        """
        Get comparison DataFrame for all experiments
        
        Args:
            metric: Metric to sort by
            
        Returns:
            Comparison DataFrame
        """
        if not self.results:
            return pd.DataFrame()
        
        metrics_dict = {name: data['metrics'] for name, data in self.results.items()}
        return compare_models(metrics_dict, metric)
    
    def get_best_model(self, metric: str = 'rmse') -> Tuple[str, Dict]:
        """
        Get the best performing model
        
        Args:
            metric: Metric to evaluate on
            
        Returns:
            (best_model_name, best_model_metrics)
        """
        if not self.results:
            return None, {}
        
        # Lower is better for all metrics except R²
        reverse = metric == 'r2'
        
        best_name = None
        best_value = -np.inf if reverse else np.inf
        
        for name, data in self.results.items():
            value = data['metrics'].get(metric, np.nan)
            
            if np.isnan(value):
                continue
            
            if reverse:
                if value > best_value:
                    best_value = value
                    best_name = name
            else:
                if value < best_value:
                    best_value = value
                    best_name = name
        
        return best_name, self.results.get(best_name, {})
    
    def save_results(self, filepath: str):
        """Save results to CSV"""
        df = self.get_comparison_df()
        df.to_csv(filepath, index=True)
        logger.info(f"Results saved to {filepath}")
    
    def print_summary(self):
        """Print summary of all experiments"""
        if not self.results:
            print("No results to display")
            return
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        print(f"Total experiments: {len(self.results)}")
        print(f"{'='*80}\n")
        
        for metric in ['mae', 'rmse', 'mape', 'r2']:
            best_name, best_data = self.get_best_model(metric)
            if best_name:
                best_value = best_data['metrics'][metric]
                print(f"Best {metric.upper():6s}: {best_name:40s} = {best_value:10.6f}")
        
        print(f"\n{'='*80}\n")
        
        # Show full comparison
        df = self.get_comparison_df('rmse')
        print("\nDetailed Comparison (sorted by RMSE):")
        print(df.to_string())


def forecast_accuracy_by_horizon(y_true: List[np.ndarray],
                                 y_pred: List[np.ndarray],
                                 horizons: List[int]) -> pd.DataFrame:
    """
    Evaluate forecast accuracy at different horizons
    
    Args:
        y_true: List of true values for each horizon
        y_pred: List of predictions for each horizon
        horizons: List of horizon values (e.g., [1, 7, 14])
        
    Returns:
        DataFrame with metrics by horizon
    """
    results = []
    
    for i, horizon in enumerate(horizons):
        metrics = compute_all_metrics(y_true[i], y_pred[i])
        metrics['horizon'] = f"{horizon}-day"
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df[['horizon'] + [col for col in df.columns if col != 'horizon']]
    
    return df


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (predicting up/down correctly)
    Useful for financial forecasting
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Directional accuracy (0-1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) < 2:
        return np.nan
    
    # Compute direction of change
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    # Calculate accuracy
    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)
    
    return correct / total if total > 0 else np.nan


def profit_loss_simulation(y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          initial_capital: float = 10000,
                          transaction_cost: float = 0.001) -> Dict:
    """
    Simulate trading based on predictions
    
    Args:
        y_true: True prices
        y_pred: Predicted prices
        initial_capital: Starting capital
        transaction_cost: Transaction cost as fraction
        
    Returns:
        Dictionary with trading simulation results
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    capital = initial_capital
    position = 0  # 0 = cash, 1 = stock
    trades = 0
    
    for i in range(1, len(y_pred)):
        # Predict direction
        if y_pred[i] > y_true[i-1]:  # Expect price to go up
            if position == 0:  # Buy
                shares = capital / y_true[i-1]
                cost = capital * transaction_cost
                capital = 0
                position = shares
                trades += 1
        elif y_pred[i] < y_true[i-1]:  # Expect price to go down
            if position > 0:  # Sell
                capital = position * y_true[i-1]
                cost = capital * transaction_cost
                capital -= cost
                position = 0
                trades += 1
    
    # Close any open position
    if position > 0:
        capital = position * y_true[-1]
        cost = capital * transaction_cost
        capital -= cost
    
    profit = capital - initial_capital
    return_pct = (profit / initial_capital) * 100
    
    return {
        'final_capital': capital,
        'profit': profit,
        'return_pct': return_pct,
        'num_trades': trades
    }

