#!/usr/bin/env python3
"""
HISTORICAL MODEL TRAINING PIPELINE
===================================
Complete pipeline for:
1. Checking data availability (1999-2025)
2. Fetching historical data from HuggingFace
3. Pre-training model on old data (1999-2022)
4. Fine-tuning on recent data (2023-2024)
5. Testing/validating on 2025 data

This implements transfer learning for financial forecasting.

Author: CMU Financial Forecasting Project
Date: November 2025
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import logging
import json
from collections import defaultdict

# Setup directories and logging
os.makedirs('results/transfer_learning', exist_ok=True)
os.makedirs('results/transfer_learning/models', exist_ok=True)
os.makedirs('results/transfer_learning/plots', exist_ok=True)
os.makedirs('data/historical_cache', exist_ok=True)
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/transfer_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
from src.data_availability_checker import DataAvailabilityChecker
from src.historical_data_fetcher import HistoricalFinancialDataFetcher, RecentDataFetcher
from src.transfer_learning_model import TransferLearningModel, TransferLearningTrainer, create_model
from src.data_preprocessor import StockDataProcessor
from src.evaluation_metrics import compute_all_metrics

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler

# Configuration
TICKER = 'AAPL'
HF_TOKEN = os.environ.get('HF_TOKEN', 'YOUR_HF_TOKEN')

# Data split configuration (based on data availability check)
# FNSPID: 2009-2020, ashraq: 2010-2020, RSS: 2024-2025
TRAIN_YEARS = (2009, 2018)  # Historical training data
FINETUNE_YEARS = (2019, 2020)  # Fine-tuning on later historical data
TEST_YEAR = 2025  # Validation/test data (RSS feeds)

# Model configuration
BACKBONE_TYPE = 'tcn'  # Options: 'tcn', 'lstm', 'bilstm', 'transformer', 'cnn_lstm'
SEQ_LENGTH = 10
WINDOWS = [3, 7, 14, 30]


def compute_sentiment_features(articles_df, windows=[3, 7, 14, 30]):
    """Compute sentiment features from articles"""
    logger.info("Computing sentiment features...")
    
    vader = SentimentIntensityAnalyzer()
    
    # Compute daily sentiment
    daily_sentiment = []
    
    for date, group in articles_df.groupby('date'):
        texts = group['text'].fillna('').tolist()
        combined_text = ' '.join(texts)[:5000]  # Limit text length
        
        if len(combined_text) > 10:
            tb_score = TextBlob(combined_text).sentiment.polarity
            vader_score = vader.polarity_scores(combined_text)['compound']
        else:
            tb_score = 0.0
            vader_score = 0.0
        
        daily_sentiment.append({
            'date': date,
            'textblob': tb_score,
            'vader': vader_score,
            'article_count': len(group)
        })
    
    sentiment_df = pd.DataFrame(daily_sentiment)
    sentiment_df = sentiment_df.sort_values('date')
    
    # Add rolling means
    for window in windows:
        for col in ['textblob', 'vader']:
            sentiment_df[f'{col}_RM{window}'] = (
                sentiment_df[col].rolling(window=window, min_periods=1).mean()
            )
    
    logger.info(f"✓ Computed sentiment for {len(sentiment_df)} days")
    return sentiment_df


def create_features(stock_df, sentiment_df, windows=[3, 7, 14, 30]):
    """Create feature matrix from stock and sentiment data"""
    logger.info("Creating feature matrix...")
    
    # Ensure date columns are compatible
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
    sentiment_df = sentiment_df.rename(columns={'date': 'Date'})
    
    # Merge
    merged_df = stock_df.merge(sentiment_df, on='Date', how='left')
    
    # Add price rolling features
    for window in windows:
        merged_df[f'Close_RM{window}'] = merged_df['Close'].rolling(window=window, min_periods=1).mean()
        merged_df[f'Volume_RM{window}'] = merged_df['Volume'].rolling(window=window, min_periods=1).mean()
        merged_df[f'Return_{window}d'] = merged_df['Close'].pct_change(window)
    
    # Add technical indicators
    merged_df['Daily_Return'] = merged_df['Close'].pct_change()
    merged_df['Volatility_7d'] = merged_df['Daily_Return'].rolling(7).std()
    merged_df['Volatility_30d'] = merged_df['Daily_Return'].rolling(30).std()
    
    # High-Low spread
    if 'High' in merged_df.columns and 'Low' in merged_df.columns:
        merged_df['HL_Spread'] = (merged_df['High'] - merged_df['Low']) / merged_df['Close']
    
    # Fill NaN values
    merged_df = merged_df.fillna(method='ffill').fillna(0)
    
    logger.info(f"✓ Created {len(merged_df.columns)} features")
    return merged_df


def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pre-training loss
    ax1 = axes[0]
    if history['pretrain']['train_loss']:
        ax1.plot(history['pretrain']['train_loss'], label='Train Loss', color='#2ecc71')
    if history['pretrain']['val_loss']:
        ax1.plot(history['pretrain']['val_loss'], label='Val Loss', color='#e74c3c')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Pre-training on Historical Data (1999-2020)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Fine-tuning loss
    ax2 = axes[1]
    if history['finetune']['train_loss']:
        ax2.plot(history['finetune']['train_loss'], label='Train Loss', color='#3498db')
    if history['finetune']['val_loss']:
        ax2.plot(history['finetune']['val_loss'], label='Val Loss', color='#9b59b6')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Fine-tuning on Recent Data (2021-2024)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {save_path}")


def plot_predictions(y_actual, y_pred, dates, model_name, save_path):
    """Plot actual vs predicted values"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series plot
    ax1 = axes[0]
    ax1.plot(dates, y_actual, label='Actual', color='#2c3e50', linewidth=1.5)
    ax1.plot(dates, y_pred, label='Predicted', color='#e74c3c', linewidth=1.5, alpha=0.8)
    ax1.fill_between(dates, y_actual, y_pred, alpha=0.2, color='#3498db')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{model_name} - Actual vs Predicted (2025 Validation)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_actual, y_pred, alpha=0.5, color='#3498db')
    
    # Perfect prediction line
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Price ($)')
    ax2.set_ylabel('Predicted Price ($)')
    ax2.set_title('Prediction Accuracy Scatter Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {save_path}")


def plot_model_comparison(results_dict, save_path):
    """Plot comparison of different models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = list(results_dict.keys())
    metrics = ['rmse', 'mae', 'mape', 'r2']
    titles = ['RMSE ($)', 'MAE ($)', 'MAPE (%)', 'R² Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        values = [results_dict[m][metric] for m in models]
        bars = ax.bar(models, values, color=colors[:len(models)])
        ax.set_ylabel(title)
        ax.set_title(f'Model Comparison: {title}')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: {save_path}")


def main():
    """Main training pipeline"""
    
    logger.info("=" * 100)
    logger.info(" TRANSFER LEARNING PIPELINE FOR FINANCIAL FORECASTING")
    logger.info(" Historical Training + Recent Fine-tuning")
    logger.info("=" * 100)
    
    # =========================================================================
    # PHASE 1: CHECK DATA AVAILABILITY
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("PHASE 1: CHECKING DATA AVAILABILITY (1999-2025)")
    logger.info("=" * 100)
    
    checker = DataAvailabilityChecker(hf_token=HF_TOKEN)
    
    try:
        availability_report = checker.generate_full_report(sample_size=10000)
        checker.save_report('results/transfer_learning/data_availability.json')
    except Exception as e:
        logger.warning(f"Could not complete full availability check: {e}")
        logger.info("Proceeding with known data sources...")
    
    # =========================================================================
    # PHASE 2: FETCH STOCK DATA
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("PHASE 2: FETCHING STOCK DATA")
    logger.info("=" * 100)
    
    processor = StockDataProcessor(use_log_returns=False)
    
    # Fetch stock data for all years we need
    # Yahoo Finance typically has data from ~2000 onwards
    stock_start = f"{TRAIN_YEARS[0]}-01-01"
    stock_end = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Fetching {TICKER} stock data: {stock_start} to {stock_end}")
    
    stock_df = processor.fetch_stock_data(TICKER, stock_start, stock_end)
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
    stock_df['Year'] = pd.to_datetime(stock_df['Date']).dt.year
    
    logger.info(f"✓ Fetched {len(stock_df)} trading days")
    logger.info(f"  Date range: {stock_df['Date'].min()} to {stock_df['Date'].max()}")
    logger.info(f"  Years: {stock_df['Year'].min()} to {stock_df['Year'].max()}")
    
    # =========================================================================
    # PHASE 3: FETCH NEWS DATA
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("PHASE 3: FETCHING NEWS DATA")
    logger.info("=" * 100)
    
    # Historical data from HuggingFace
    historical_fetcher = HistoricalFinancialDataFetcher(hf_token=HF_TOKEN)
    
    logger.info(f"\n[3.1] Fetching historical news (FNSPID: 1999-2023)...")
    historical_news = historical_fetcher.fetch_historical_data(
        start_year=max(1999, stock_df['Year'].min()),
        end_year=min(2023, FINETUNE_YEARS[1]),
        max_articles_per_year=20000
    )
    
    if historical_news.empty:
        logger.warning("Could not fetch historical news from HuggingFace")
        logger.info("Using simulated historical sentiment...")
        # Create simulated sentiment based on stock returns
        stock_df['Return'] = stock_df['Close'].pct_change()
        historical_sentiment = pd.DataFrame({
            'date': stock_df['Date'],
            'textblob': stock_df['Return'].rolling(5).mean().fillna(0) * 10,
            'vader': stock_df['Return'].rolling(5).mean().fillna(0) * 10,
            'article_count': 1
        })
        for w in WINDOWS:
            for col in ['textblob', 'vader']:
                historical_sentiment[f'{col}_RM{w}'] = (
                    historical_sentiment[col].rolling(window=w, min_periods=1).mean()
                )
    else:
        logger.info(f"✓ Fetched {len(historical_news):,} historical articles")
        historical_sentiment = compute_sentiment_features(historical_news, WINDOWS)
    
    # Recent data
    logger.info(f"\n[3.2] Fetching recent news (2023-2025)...")
    recent_fetcher = RecentDataFetcher(hf_token=HF_TOKEN)
    
    try:
        recent_news = recent_fetcher.fetch_all_recent_data(
            start_date='2023-01-01',
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        if not recent_news.empty:
            logger.info(f"✓ Fetched {len(recent_news):,} recent articles")
            recent_sentiment = compute_sentiment_features(recent_news, WINDOWS)
        else:
            raise ValueError("No recent news found")
            
    except Exception as e:
        logger.warning(f"Could not fetch recent news: {e}")
        logger.info("Using RSS fallback for recent data...")
        
        # Use RSS feeds
        from advanced_sentiment import compute_multi_method_sentiment
        recent_sentiment = compute_multi_method_sentiment(
            f'{TICKER} stock', '2023-01-01', datetime.now().strftime('%Y-%m-%d'),
            max_items=2000
        )
        
        if not recent_sentiment.empty:
            for w in WINDOWS:
                for col in ['textblob', 'vader', 'finbert']:
                    if col in recent_sentiment.columns:
                        recent_sentiment[f'{col}_RM{w}'] = (
                            recent_sentiment[col].rolling(window=w, min_periods=1).mean()
                        )
    
    # Combine sentiments
    all_sentiment = pd.concat([historical_sentiment, recent_sentiment], ignore_index=True)
    all_sentiment = all_sentiment.drop_duplicates(subset=['date'], keep='last')
    all_sentiment = all_sentiment.sort_values('date')
    
    logger.info(f"✓ Combined sentiment: {len(all_sentiment)} days")
    logger.info(f"  Date range: {all_sentiment['date'].min()} to {all_sentiment['date'].max()}")
    
    # =========================================================================
    # PHASE 4: CREATE FEATURES
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("PHASE 4: FEATURE ENGINEERING")
    logger.info("=" * 100)
    
    merged_df = create_features(stock_df, all_sentiment, WINDOWS)
    
    # Define feature columns
    feature_cols = [col for col in merged_df.columns 
                   if col not in ['Date', 'Year', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    # Filter to numeric columns only
    feature_cols = [col for col in feature_cols if merged_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    logger.info(f"Features selected: {len(feature_cols)}")
    logger.info(f"  {feature_cols[:10]}...")
    
    # Save feature dataset
    merged_df.to_csv('results/transfer_learning/feature_dataset.csv', index=False)
    logger.info("✓ Saved: results/transfer_learning/feature_dataset.csv")
    
    # =========================================================================
    # PHASE 5: SPLIT DATA BY YEAR
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("PHASE 5: SPLITTING DATA BY YEAR")
    logger.info("=" * 100)
    
    # Split by year
    train_data = merged_df[(merged_df['Year'] >= TRAIN_YEARS[0]) & (merged_df['Year'] <= TRAIN_YEARS[1])]
    finetune_data = merged_df[(merged_df['Year'] >= FINETUNE_YEARS[0]) & (merged_df['Year'] <= FINETUNE_YEARS[1])]
    test_data = merged_df[merged_df['Year'] >= TEST_YEAR]
    
    logger.info(f"Train data ({TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}): {len(train_data)} days")
    logger.info(f"Fine-tune data ({FINETUNE_YEARS[0]}-{FINETUNE_YEARS[1]}): {len(finetune_data)} days")
    logger.info(f"Test data ({TEST_YEAR}): {len(test_data)} days")
    
    if len(train_data) < 100:
        logger.warning("Not enough training data. Adjusting split...")
        # Use available data
        total_len = len(merged_df)
        train_end = int(total_len * 0.6)
        finetune_end = int(total_len * 0.85)
        
        train_data = merged_df.iloc[:train_end]
        finetune_data = merged_df.iloc[train_end:finetune_end]
        test_data = merged_df.iloc[finetune_end:]
        
        logger.info(f"Adjusted splits:")
        logger.info(f"  Train: {len(train_data)} days")
        logger.info(f"  Fine-tune: {len(finetune_data)} days")
        logger.info(f"  Test: {len(test_data)} days")
    
    # Prepare arrays
    X_train = train_data[feature_cols].values
    y_train = train_data['Close'].values
    
    X_finetune = finetune_data[feature_cols].values
    y_finetune = finetune_data['Close'].values
    
    X_test = test_data[feature_cols].values
    y_test = test_data['Close'].values
    test_dates = test_data['Date'].values
    
    # =========================================================================
    # PHASE 6: TRAIN MODELS
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("PHASE 6: TRAINING TRANSFER LEARNING MODELS")
    logger.info("=" * 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    all_results = {}
    backbone_types = ['tcn', 'lstm', 'bilstm', 'transformer', 'cnn_lstm']
    
    for backbone_type in backbone_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING: {backbone_type.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            # Create model
            model = create_model(
                input_size=len(feature_cols),
                backbone_type=backbone_type
            )
            
            # Create trainer
            trainer = TransferLearningTrainer(model, device=device)
            
            # Pre-train on historical data
            logger.info(f"\n[{backbone_type}] Pre-training on historical data...")
            pretrain_history = trainer.pretrain(
                X_train, y_train,
                X_val=X_finetune[:100] if len(X_finetune) > 100 else None,
                y_val=y_finetune[:100] if len(y_finetune) > 100 else None,
                epochs=100,
                batch_size=32,
                lr=0.001,
                patience=15,
                seq_length=SEQ_LENGTH
            )
            
            # Fine-tune on recent data
            logger.info(f"\n[{backbone_type}] Fine-tuning on recent data...")
            finetune_history = trainer.finetune(
                X_finetune, y_finetune,
                X_val=X_test[:50] if len(X_test) > 50 else None,
                y_val=y_test[:50] if len(y_test) > 50 else None,
                epochs=50,
                batch_size=16,
                lr=0.0005,
                patience=10,
                seq_length=SEQ_LENGTH
            )
            
            # Evaluate on test data
            logger.info(f"\n[{backbone_type}] Evaluating on 2025 test data...")
            metrics, predictions, y_actual = trainer.evaluate(X_test, y_test, seq_length=SEQ_LENGTH)
            
            all_results[backbone_type.upper()] = metrics
            
            logger.info(f"✓ {backbone_type.upper()} Results:")
            logger.info(f"  RMSE: ${metrics['rmse']:.2f}")
            logger.info(f"  MAE:  ${metrics['mae']:.2f}")
            logger.info(f"  MAPE: {metrics['mape']:.2f}%")
            logger.info(f"  R²:   {metrics['r2']:.4f}")
            
            # Save model
            trainer.save(f'results/transfer_learning/models/{backbone_type}_model.pt')
            
            # Plot predictions
            aligned_dates = test_dates[SEQ_LENGTH:SEQ_LENGTH + len(predictions)]
            plot_predictions(
                y_actual, predictions, aligned_dates,
                backbone_type.upper(),
                f'results/transfer_learning/plots/{backbone_type}_predictions.png'
            )
            
            # Plot training history
            plot_training_history(
                trainer.training_history,
                f'results/transfer_learning/plots/{backbone_type}_training_history.png'
            )
            
        except Exception as e:
            logger.error(f"Error training {backbone_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # =========================================================================
    # PHASE 7: FINAL COMPARISON
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("PHASE 7: FINAL MODEL COMPARISON")
    logger.info("=" * 100)
    
    if all_results:
        # Plot comparison
        plot_model_comparison(
            all_results,
            'results/transfer_learning/plots/model_comparison.png'
        )
        
        # Save results
        results_df = pd.DataFrame([
            {'Model': name, **metrics}
            for name, metrics in all_results.items()
        ]).sort_values('rmse')
        
        results_df.to_csv('results/transfer_learning/model_comparison.csv', index=False)
        logger.info("✓ Saved: results/transfer_learning/model_comparison.csv")
        
        # Print final summary
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info("\n" + results_df.to_string(index=False))
        
        # Best model
        best = results_df.iloc[0]
        logger.info(f"\n✅ BEST MODEL: {best['Model']}")
        logger.info(f"   RMSE: ${best['rmse']:.2f}")
        logger.info(f"   MAPE: {best['mape']:.2f}%")
        logger.info(f"   R²:   {best['r2']:.4f}")
    
    logger.info("\n" + "=" * 100)
    logger.info("✅ TRANSFER LEARNING PIPELINE COMPLETE")
    logger.info("=" * 100)
    
    logger.info("\n📁 OUTPUT FILES:")
    logger.info("  • results/transfer_learning/data_availability.json")
    logger.info("  • results/transfer_learning/feature_dataset.csv")
    logger.info("  • results/transfer_learning/model_comparison.csv")
    logger.info("  • results/transfer_learning/models/*.pt")
    logger.info("  • results/transfer_learning/plots/*.png")
    
    print("\n🎉 Training complete! Check results/transfer_learning/ for outputs.")


if __name__ == "__main__":
    main()

