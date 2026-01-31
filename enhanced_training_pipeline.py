#!/usr/bin/env python3
"""
ENHANCED TRAINING PIPELINE - PROFESSOR'S REQUIREMENTS
======================================================
Implements all professor's feedback:
1. Case study section for market events with sentiment weights
2. Correct inverse-scaling matching normalization factor
3. Non-Apple news: govt, supply-chain, Cupertino, macro events
4. Fixed rolling-prediction logic with full window
5. Transformer variants experimentation
6. Single-layer linear baseline
7. SARIMAX (1,1,1) benchmark without seasonal terms

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
import torch.nn as nn
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict

# Setup
os.makedirs('results/enhanced_v2', exist_ok=True)
os.makedirs('results/enhanced_v2/case_studies', exist_ok=True)
os.makedirs('results/enhanced_v2/models', exist_ok=True)
os.makedirs('results/enhanced_v2/plots', exist_ok=True)
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_training_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from src.data_preprocessor import StockDataProcessor
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configuration
TICKER = 'AAPL'
HF_TOKEN = os.environ.get('HF_TOKEN', 'YOUR_HF_TOKEN')
WINDOWS = [3, 7, 14, 30]
SEQ_LENGTH = 10


# =============================================================================
# 1. NON-APPLE NEWS FETCHER - Broader Market Headlines
# =============================================================================
class BroadMarketNewsFetcher:
    """
    Fetches news that could indirectly affect Apple stock:
    - Government/elections
    - Supply chain
    - Cupertino local events
    - Broader market/macro headlines
    - Tech sector news
    """
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        # Keywords that could affect AAPL indirectly
        self.indirect_keywords = {
            'government': ['election', 'congress', 'fed', 'federal reserve', 'interest rate', 
                          'tariff', 'trade war', 'regulation', 'antitrust', 'biden', 'trump',
                          'tax', 'policy', 'legislation'],
            'supply_chain': ['chip shortage', 'semiconductor', 'foxconn', 'tsmc', 'supply chain',
                            'shipping', 'logistics', 'manufacturing', 'component', 'shortage'],
            'cupertino': ['cupertino', 'silicon valley', 'bay area', 'california', 'tech hub'],
            'macro': ['inflation', 'gdp', 'unemployment', 'recession', 'market crash',
                     'stock market', 'nasdaq', 'sp500', 'dow jones', 'economic'],
            'tech_sector': ['google', 'microsoft', 'amazon', 'meta', 'facebook', 'samsung',
                           'huawei', 'xiaomi', 'tech stocks', 'big tech']
        }
    
    def classify_news_type(self, text):
        """Classify news into categories"""
        text_lower = text.lower()
        categories = []
        
        for category, keywords in self.indirect_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    categories.append(category)
                    break
        
        return categories if categories else ['general']
    
    def compute_weighted_sentiment(self, text, categories):
        """Compute sentiment with category-based weights"""
        base_sentiment = self.vader.polarity_scores(text)['compound']
        
        # Weight multipliers based on likely impact on AAPL
        weights = {
            'government': 0.7,
            'supply_chain': 1.2,  # High impact
            'cupertino': 0.8,
            'macro': 0.9,
            'tech_sector': 1.0,
            'general': 0.5
        }
        
        if not categories:
            return base_sentiment * 0.5
        
        avg_weight = np.mean([weights.get(cat, 0.5) for cat in categories])
        return base_sentiment * avg_weight
    
    def fetch_broader_news(self, days_back=365):
        """Fetch broader market news from RSS"""
        import feedparser
        
        logger.info("Fetching broader market news (non-Apple specific)...")
        
        queries = [
            'federal reserve interest rate',
            'semiconductor chip shortage',
            'tech stock market',
            'us economy inflation',
            'supply chain disruption'
        ]
        
        all_articles = []
        
        for query in queries:
            try:
                url = f'https://news.google.com/rss/search?q={query.replace(" ", "+")}&hl=en-US'
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:20]:
                    try:
                        pub_date = pd.to_datetime(entry.published)
                        categories = self.classify_news_type(entry.title)
                        sentiment = self.compute_weighted_sentiment(entry.title, categories)
                        
                        all_articles.append({
                            'date': pub_date.date(),
                            'title': entry.title,
                            'categories': categories,
                            'sentiment': sentiment,
                            'query': query
                        })
                    except:
                        continue
                        
            except Exception as e:
                logger.warning(f"Error fetching {query}: {e}")
        
        df = pd.DataFrame(all_articles)
        if not df.empty:
            df = df.drop_duplicates(subset=['date', 'title'])
            logger.info(f"✓ Fetched {len(df)} broader market articles")
        
        return df


# =============================================================================
# 2. CASE STUDY ANALYZER - Market Event Analysis
# =============================================================================
class CaseStudyAnalyzer:
    """
    Analyzes specific market events with:
    - Headline words
    - Sentiment weights
    - Model's predicted price move
    """
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        # Major market events to analyze
        self.major_events = {
            '2020-03-16': {
                'name': 'COVID-19 Market Crash',
                'description': 'Market circuit breaker triggered, massive selloff'
            },
            '2020-03-23': {
                'name': 'COVID-19 Market Bottom',
                'description': 'Fed announces unlimited QE, market bottoms'
            },
            '2022-01-03': {
                'name': 'AAPL $3T Market Cap',
                'description': 'Apple becomes first $3 trillion company'
            },
            '2022-06-13': {
                'name': 'Bear Market 2022',
                'description': 'S&P 500 enters bear market, tech selloff'
            },
            '2023-08-03': {
                'name': 'AAPL Q3 2023 Earnings',
                'description': 'Revenue decline but Services strong'
            }
        }
    
    def analyze_headline(self, headline):
        """Break down headline into words with sentiment weights"""
        words = headline.lower().split()
        
        word_sentiments = []
        for word in words:
            if len(word) > 2:
                score = self.vader.polarity_scores(word)['compound']
                word_sentiments.append({
                    'word': word,
                    'sentiment': score,
                    'contribution': abs(score) > 0.1
                })
        
        return word_sentiments
    
    def generate_case_study(self, date, headlines, actual_price, predicted_price, 
                           prev_price, model_name='Model'):
        """Generate detailed case study for a date"""
        
        actual_move = (actual_price - prev_price) / prev_price * 100
        predicted_move = (predicted_price - prev_price) / prev_price * 100
        
        case_study = {
            'date': str(date),
            'event_info': self.major_events.get(str(date), {'name': 'Trading Day', 'description': ''}),
            'headlines': [],
            'actual_price': actual_price,
            'predicted_price': predicted_price,
            'prev_price': prev_price,
            'actual_move_pct': actual_move,
            'predicted_move_pct': predicted_move,
            'prediction_error_pct': predicted_move - actual_move,
            'model': model_name
        }
        
        for headline in headlines[:5]:  # Top 5 headlines
            headline_analysis = {
                'text': headline,
                'overall_sentiment': self.vader.polarity_scores(headline)['compound'],
                'word_breakdown': self.analyze_headline(headline)
            }
            case_study['headlines'].append(headline_analysis)
        
        return case_study
    
    def plot_case_study(self, case_study, save_path):
        """Visualize case study"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Price comparison
        ax1 = axes[0, 0]
        prices = [case_study['prev_price'], case_study['actual_price'], case_study['predicted_price']]
        labels = ['Previous', 'Actual', 'Predicted']
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        ax1.bar(labels, prices, color=colors)
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f"Price Comparison: {case_study['date']}")
        
        # 2. Move comparison
        ax2 = axes[0, 1]
        moves = [case_study['actual_move_pct'], case_study['predicted_move_pct']]
        ax2.bar(['Actual Move', 'Predicted Move'], moves, 
               color=['#2ecc71', '#e74c3c'])
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Price Move (%)')
        ax2.set_title('Actual vs Predicted Move')
        
        # 3. Headline sentiment
        ax3 = axes[1, 0]
        if case_study['headlines']:
            sentiments = [h['overall_sentiment'] for h in case_study['headlines']]
            indices = range(len(sentiments))
            colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in sentiments]
            ax3.barh(indices, sentiments, color=colors)
            ax3.set_xlabel('Sentiment Score')
            ax3.set_ylabel('Headline #')
            ax3.set_title('Headline Sentiments')
            ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # 4. Word cloud / key words
        ax4 = axes[1, 1]
        if case_study['headlines']:
            all_words = []
            for h in case_study['headlines']:
                for w in h['word_breakdown']:
                    if w['contribution']:
                        all_words.append((w['word'], w['sentiment']))
            
            if all_words:
                words, sents = zip(*all_words[:15])
                colors = ['#2ecc71' if s > 0 else '#e74c3c' if s < 0 else '#95a5a6' for s in sents]
                ax4.barh(range(len(words)), sents, color=colors)
                ax4.set_yticks(range(len(words)))
                ax4.set_yticklabels(words)
                ax4.set_xlabel('Sentiment Weight')
                ax4.set_title('Key Words & Sentiment Weights')
        
        plt.suptitle(f"Case Study: {case_study['event_info']['name']}\n{case_study['date']}", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved case study: {save_path}")


# =============================================================================
# 3. CORRECT SCALING CLASS - Proper Normalization/Denormalization
# =============================================================================
class CorrectScaler:
    """
    Ensures inverse-scaling uses EXACTLY the same factor as normalization
    to prevent price jumps and discontinuities.
    """
    
    def __init__(self):
        self.scale_factor = None
        self.min_val = None
        self.is_fitted = False
    
    def fit(self, data):
        """Fit scaler and store EXACT parameters"""
        data = np.array(data).flatten()
        self.min_val = float(np.min(data))
        self.max_val = float(np.max(data))
        self.scale_factor = float(self.max_val - self.min_val)
        
        if self.scale_factor == 0:
            self.scale_factor = 1.0
        
        self.is_fitted = True
        
        logger.info(f"Scaler fitted: min={self.min_val:.4f}, max={self.max_val:.4f}, scale={self.scale_factor:.4f}")
        return self
    
    def transform(self, data):
        """Transform data using stored parameters"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted!")
        
        data = np.array(data)
        return (data - self.min_val) / self.scale_factor
    
    def inverse_transform(self, scaled_data):
        """
        Inverse transform using EXACT same factors
        Critical: Uses stored scale_factor to prevent discontinuities
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted!")
        
        scaled_data = np.array(scaled_data)
        
        # Use EXACT same factors
        return scaled_data * self.scale_factor + self.min_val
    
    def fit_transform(self, data):
        """Fit and transform"""
        self.fit(data)
        return self.transform(data)


# =============================================================================
# 4. ROLLING PREDICTION WITH FULL WINDOW UPDATE
# =============================================================================
class RollingPredictor:
    """
    Implements correct rolling prediction:
    - Updates with FULL window after each day
    - Not just a single new point
    """
    
    def __init__(self, model, scaler_X, scaler_y, seq_length=10, device='cpu'):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.seq_length = seq_length
        self.device = device
    
    def predict_rolling(self, X_data, y_actual):
        """
        Rolling prediction with full window update
        
        For each prediction:
        1. Use full window of seq_length days
        2. Make prediction
        3. Shift window forward by 1 day
        4. Update window with actual new data
        """
        self.model.eval()
        
        predictions = []
        n_samples = len(X_data)
        
        # Scale all data
        X_scaled = self.scaler_X.transform(X_data)
        
        logger.info(f"Rolling prediction: {n_samples - self.seq_length} predictions with window={self.seq_length}")
        
        for i in range(self.seq_length, n_samples):
            # Get FULL window of seq_length days
            window_start = i - self.seq_length
            window_end = i
            
            # Full window with all actual values up to this point
            X_window = X_scaled[window_start:window_end]
            
            # Reshape for model: (batch=1, seq_length, features)
            X_tensor = torch.FloatTensor(X_window).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred_scaled = self.model(X_tensor).cpu().numpy().flatten()[0]
            
            # Inverse transform using EXACT scaler
            pred = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(pred)
        
        return np.array(predictions)


# =============================================================================
# 5. MODEL ARCHITECTURES
# =============================================================================

class SingleLayerLinear(nn.Module):
    """Single-layer linear network with linear activation for baseline"""
    
    def __init__(self, input_size, seq_length):
        super().__init__()
        self.flatten_size = input_size * seq_length
        self.linear = nn.Linear(self.flatten_size, 1)
    
    def forward(self, x):
        # x: (batch, seq, features)
        x = x.contiguous().view(x.size(0), -1)  # Flatten (use contiguous)
        return self.linear(x)  # Linear activation (no non-linearity)


class TransformerVariant(nn.Module):
    """Transformer with configurable heads and dimensions"""
    
    def __init__(self, input_size, d_model=32, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.output(x[:, -1, :])


class TCNModel(nn.Module):
    """TCN for comparison"""
    
    def __init__(self, input_size, hidden_channels=[32, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        
        layers = []
        num_channels = [input_size] + hidden_channels
        
        for i in range(len(hidden_channels)):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            
            layers.append(nn.Conv1d(num_channels[i], num_channels[i+1], 
                                   kernel_size, dilation=dilation, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_channels[-1], 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = self.network(x)
        return self.output(x[:, :, -1])


class LSTMModel(nn.Module):
    """LSTM for comparison"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.output(out[:, -1, :])


# =============================================================================
# 6. TRAINING WITH PROPER VALIDATION SPLIT
# =============================================================================
class EnhancedTrainer:
    """
    Training with:
    - Proper train/val/test split
    - Correct epoch loops
    - Early stopping
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.training_history = defaultdict(list)
    
    def prepare_sequences(self, X, y, seq_length):
        """Create sequences for time series"""
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    def train_model(self, model, X_train, y_train, X_val, y_val,
                   epochs=100, batch_size=32, lr=0.001, patience=15,
                   seq_length=10, model_name='Model'):
        """Train with proper validation"""
        
        # Use correct scaler
        scaler_X = CorrectScaler()
        scaler_y = CorrectScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        X_val_scaled = scaler_X.transform(X_val)
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train_scaled, y_train_scaled, seq_length)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val_scaled, seq_length)
        
        logger.info(f"Training {model_name}: {len(X_train_seq)} train, {len(X_val_seq)} val samples")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            model.train()
            
            permutation = torch.randperm(X_train_tensor.size(0))
            epoch_train_loss = 0
            n_batches = 0
            
            for i in range(0, X_train_tensor.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_X = X_train_tensor[indices]
                batch_y = y_train_tensor[indices]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_train_loss / n_batches
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            self.training_history[f'{model_name}_train'].append(avg_train_loss)
            self.training_history[f'{model_name}_val'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"  Epoch {epoch+1}: Train={avg_train_loss:.6f}, Val={val_loss:.6f}")
        
        # Restore best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return model, scaler_X, scaler_y


# =============================================================================
# 7. SARIMAX BENCHMARK
# =============================================================================
def train_sarimax_benchmark(train_data, test_data, exog_train=None, exog_test=None):
    """
    SARIMAX with differencing=1, no seasonal terms
    Reference benchmark
    """
    logger.info("Training SARIMAX benchmark (order=(1,1,1), no seasonal)...")
    
    history = list(train_data)
    if exog_train is not None:
        history_exog = list(exog_train)
    
    predictions = []
    
    for t in range(len(test_data)):
        try:
            if exog_train is not None:
                model = SARIMAX(
                    history,
                    exog=np.array(history_exog).reshape(-1, 1) if exog_train is not None else None,
                    order=(1, 1, 1),  # Differencing order 1
                    seasonal_order=(0, 0, 0, 0),  # No seasonal terms
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_fit = model.fit(disp=False, maxiter=100)
                yhat = model_fit.forecast(
                    steps=1, 
                    exog=exog_test[t].reshape(1, -1) if exog_test is not None else None
                )[0]
            else:
                model = SARIMAX(
                    history,
                    order=(1, 1, 1),
                    seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_fit = model.fit(disp=False, maxiter=100)
                yhat = model_fit.forecast(steps=1)[0]
        except:
            yhat = history[-1]
        
        predictions.append(yhat)
        history.append(test_data[t])
        if exog_train is not None:
            history_exog.append(exog_test[t])
        
        if (t + 1) % 50 == 0:
            logger.info(f"  SARIMAX: {t+1}/{len(test_data)} predictions")
    
    return np.array(predictions)


# =============================================================================
# 8. COMPUTE METRICS
# =============================================================================
def compute_metrics(y_true, y_pred):
    """Compute comprehensive metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    logger.info("=" * 100)
    logger.info(" ENHANCED TRAINING PIPELINE - ALL PROFESSOR'S REQUIREMENTS")
    logger.info("=" * 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # =========================================================================
    # FETCH DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: DATA COLLECTION")
    logger.info("=" * 80)
    
    # Stock data
    processor = StockDataProcessor(use_log_returns=False)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3 years
    
    stock_df = processor.fetch_stock_data(
        TICKER, 
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
    
    logger.info(f"✓ Fetched {len(stock_df)} trading days for {TICKER}")
    
    # Broader market news (non-Apple)
    broader_fetcher = BroadMarketNewsFetcher()
    broader_news = broader_fetcher.fetch_broader_news(days_back=365)
    
    # Apple-specific news via RSS
    logger.info("Fetching Apple-specific news...")
    import feedparser
    
    apple_articles = []
    feed = feedparser.parse('https://news.google.com/rss/search?q=apple+stock&hl=en-US')
    for entry in feed.entries[:100]:
        try:
            pub_date = pd.to_datetime(entry.published)
            apple_articles.append({
                'date': pub_date.date(),
                'title': entry.title,
                'categories': ['apple_direct'],
                'sentiment': SentimentIntensityAnalyzer().polarity_scores(entry.title)['compound']
            })
        except:
            continue
    
    apple_news = pd.DataFrame(apple_articles)
    logger.info(f"✓ Fetched {len(apple_news)} Apple-specific articles")
    
    # Combine news
    all_news = pd.concat([broader_news, apple_news], ignore_index=True) if not broader_news.empty else apple_news
    all_news = all_news.drop_duplicates(subset=['date', 'title'])
    logger.info(f"✓ Combined: {len(all_news)} total articles")
    
    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    # Aggregate sentiment by date
    daily_sentiment = all_news.groupby('date').agg({
        'sentiment': 'mean',
        'title': 'count'
    }).rename(columns={'title': 'article_count'})
    daily_sentiment.reset_index(inplace=True)
    daily_sentiment.columns = ['Date', 'sentiment', 'article_count']
    
    # Add rolling features
    for window in WINDOWS:
        stock_df[f'Close_RM{window}'] = stock_df['Close'].rolling(window, min_periods=1).mean()
        stock_df[f'Volume_RM{window}'] = stock_df['Volume'].rolling(window, min_periods=1).mean()
        stock_df[f'Return_{window}d'] = stock_df['Close'].pct_change(window)
    
    stock_df['Daily_Return'] = stock_df['Close'].pct_change()
    stock_df['Volatility_7d'] = stock_df['Daily_Return'].rolling(7).std()
    
    # Merge sentiment
    merged_df = stock_df.merge(daily_sentiment, on='Date', how='left')
    merged_df['sentiment'] = merged_df['sentiment'].fillna(0)
    merged_df['article_count'] = merged_df['article_count'].fillna(0)
    
    # Add sentiment rolling means
    for window in WINDOWS:
        merged_df[f'sentiment_RM{window}'] = merged_df['sentiment'].rolling(window, min_periods=1).mean()
    
    merged_df = merged_df.fillna(method='ffill').fillna(0)
    
    # Feature columns
    feature_cols = [col for col in merged_df.columns 
                   if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    feature_cols = [col for col in feature_cols 
                   if merged_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    logger.info(f"✓ Created {len(feature_cols)} features")
    
    # Save dataset
    merged_df.to_csv('results/enhanced_v2/feature_dataset.csv', index=False)
    
    # =========================================================================
    # TRAIN/VAL/TEST SPLIT
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: DATA SPLITTING")
    logger.info("=" * 80)
    
    total_len = len(merged_df)
    train_end = int(total_len * 0.6)
    val_end = int(total_len * 0.8)
    
    train_df = merged_df.iloc[:train_end]
    val_df = merged_df.iloc[train_end:val_end]
    test_df = merged_df.iloc[val_end:]
    
    logger.info(f"Train: {len(train_df)} days ({merged_df['Date'].iloc[0]} to {train_df['Date'].iloc[-1]})")
    logger.info(f"Val:   {len(val_df)} days ({val_df['Date'].iloc[0]} to {val_df['Date'].iloc[-1]})")
    logger.info(f"Test:  {len(test_df)} days ({test_df['Date'].iloc[0]} to {merged_df['Date'].iloc[-1]})")
    
    X_train = train_df[feature_cols].values
    y_train = train_df['Close'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['Close'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['Close'].values
    test_dates = test_df['Date'].values
    
    # =========================================================================
    # MODEL TRAINING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: MODEL TRAINING")
    logger.info("=" * 80)
    
    trainer = EnhancedTrainer(device=device)
    all_results = {}
    
    # 1. SARIMAX Benchmark (order=1,1,1, no seasonal)
    logger.info("\n[Model 1] SARIMAX (1,1,1) Benchmark...")
    sarimax_pred = train_sarimax_benchmark(
        y_train, y_test,
        exog_train=merged_df[[f'sentiment_RM{WINDOWS[1]}']].iloc[:train_end].values,
        exog_test=merged_df[[f'sentiment_RM{WINDOWS[1]}']].iloc[val_end:].values
    )
    all_results['SARIMAX(1,1,1)'] = compute_metrics(y_test, sarimax_pred)
    logger.info(f"  RMSE: ${all_results['SARIMAX(1,1,1)']['rmse']:.2f}")
    
    # 2. Single-Layer Linear (linear activation)
    logger.info("\n[Model 2] Single-Layer Linear Network...")
    linear_model = SingleLayerLinear(len(feature_cols), SEQ_LENGTH).to(device)
    linear_model, linear_scaler_X, linear_scaler_y = trainer.train_model(
        linear_model, X_train, y_train, X_val, y_val,
        epochs=100, model_name='LinearNet'
    )
    
    # Rolling prediction for linear
    linear_predictor = RollingPredictor(linear_model, linear_scaler_X, linear_scaler_y, SEQ_LENGTH, device)
    linear_pred = linear_predictor.predict_rolling(
        np.vstack([X_train[-SEQ_LENGTH:], X_val, X_test]), 
        np.concatenate([y_train[-SEQ_LENGTH:], y_val, y_test])
    )[-len(y_test):]
    all_results['LinearNet'] = compute_metrics(y_test, linear_pred)
    logger.info(f"  RMSE: ${all_results['LinearNet']['rmse']:.2f}")
    
    # 3. LSTM
    logger.info("\n[Model 3] LSTM...")
    lstm_model = LSTMModel(len(feature_cols)).to(device)
    lstm_model, lstm_scaler_X, lstm_scaler_y = trainer.train_model(
        lstm_model, X_train, y_train, X_val, y_val,
        epochs=100, model_name='LSTM'
    )
    lstm_predictor = RollingPredictor(lstm_model, lstm_scaler_X, lstm_scaler_y, SEQ_LENGTH, device)
    lstm_pred = lstm_predictor.predict_rolling(
        np.vstack([X_train[-SEQ_LENGTH:], X_val, X_test]),
        np.concatenate([y_train[-SEQ_LENGTH:], y_val, y_test])
    )[-len(y_test):]
    all_results['LSTM'] = compute_metrics(y_test, lstm_pred)
    logger.info(f"  RMSE: ${all_results['LSTM']['rmse']:.2f}")
    
    # 4. TCN
    logger.info("\n[Model 4] TCN...")
    tcn_model = TCNModel(len(feature_cols)).to(device)
    tcn_model, tcn_scaler_X, tcn_scaler_y = trainer.train_model(
        tcn_model, X_train, y_train, X_val, y_val,
        epochs=100, model_name='TCN'
    )
    tcn_predictor = RollingPredictor(tcn_model, tcn_scaler_X, tcn_scaler_y, SEQ_LENGTH, device)
    tcn_pred = tcn_predictor.predict_rolling(
        np.vstack([X_train[-SEQ_LENGTH:], X_val, X_test]),
        np.concatenate([y_train[-SEQ_LENGTH:], y_val, y_test])
    )[-len(y_test):]
    all_results['TCN'] = compute_metrics(y_test, tcn_pred)
    logger.info(f"  RMSE: ${all_results['TCN']['rmse']:.2f}")
    
    # 5. Transformer Variants
    transformer_configs = [
        {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'name': 'Transformer-64d-4h-2L'},
        {'d_model': 32, 'nhead': 2, 'num_layers': 1, 'name': 'Transformer-32d-2h-1L'},
        {'d_model': 16, 'nhead': 2, 'num_layers': 1, 'name': 'Transformer-16d-2h-1L'},
    ]
    
    for config in transformer_configs:
        logger.info(f"\n[Model] {config['name']}...")
        tf_model = TransformerVariant(
            len(feature_cols), 
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers']
        ).to(device)
        
        tf_model, tf_scaler_X, tf_scaler_y = trainer.train_model(
            tf_model, X_train, y_train, X_val, y_val,
            epochs=100, model_name=config['name']
        )
        
        tf_predictor = RollingPredictor(tf_model, tf_scaler_X, tf_scaler_y, SEQ_LENGTH, device)
        tf_pred = tf_predictor.predict_rolling(
            np.vstack([X_train[-SEQ_LENGTH:], X_val, X_test]),
            np.concatenate([y_train[-SEQ_LENGTH:], y_val, y_test])
        )[-len(y_test):]
        all_results[config['name']] = compute_metrics(y_test, tf_pred)
        logger.info(f"  RMSE: ${all_results[config['name']]['rmse']:.2f}")
    
    # =========================================================================
    # CASE STUDIES
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: CASE STUDIES")
    logger.info("=" * 80)
    
    case_analyzer = CaseStudyAnalyzer()
    
    # Find interesting dates in test set
    test_returns = np.diff(y_test) / y_test[:-1] * 100
    interesting_indices = np.where(np.abs(test_returns) > 2)[0]  # Days with >2% move
    
    if len(interesting_indices) > 0:
        for idx in interesting_indices[:3]:  # Top 3 interesting days
            date = test_dates[idx + 1]
            
            # Get headlines for this date
            date_news = all_news[all_news['date'] == date]
            headlines = date_news['title'].tolist() if not date_news.empty else ['No headlines available']
            
            case_study = case_analyzer.generate_case_study(
                date=date,
                headlines=headlines,
                actual_price=y_test[idx + 1],
                predicted_price=lstm_pred[idx + 1] if idx + 1 < len(lstm_pred) else y_test[idx],
                prev_price=y_test[idx],
                model_name='LSTM'
            )
            
            case_analyzer.plot_case_study(
                case_study,
                f'results/enhanced_v2/case_studies/case_study_{str(date)}.png'
            )
            
            # Save case study data
            pd.DataFrame([case_study]).to_json(
                f'results/enhanced_v2/case_studies/case_study_{str(date)}.json'
            )
    
    # =========================================================================
    # RESULTS COMPARISON
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6: RESULTS COMPARISON")
    logger.info("=" * 80)
    
    results_df = pd.DataFrame([
        {'Model': name, **metrics}
        for name, metrics in all_results.items()
    ]).sort_values('rmse')
    
    results_df.to_csv('results/enhanced_v2/model_comparison.csv', index=False)
    
    logger.info("\n" + results_df.to_string(index=False))
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = results_df['Model'].tolist()
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    # RMSE
    axes[0, 0].barh(models, results_df['rmse'], color=colors)
    axes[0, 0].set_xlabel('RMSE ($)')
    axes[0, 0].set_title('Model Comparison: RMSE')
    
    # MAE
    axes[0, 1].barh(models, results_df['mae'], color=colors)
    axes[0, 1].set_xlabel('MAE ($)')
    axes[0, 1].set_title('Model Comparison: MAE')
    
    # MAPE
    axes[1, 0].barh(models, results_df['mape'], color=colors)
    axes[1, 0].set_xlabel('MAPE (%)')
    axes[1, 0].set_title('Model Comparison: MAPE')
    
    # R²
    axes[1, 1].barh(models, results_df['r2'], color=colors)
    axes[1, 1].set_xlabel('R² Score')
    axes[1, 1].set_title('Model Comparison: R²')
    
    plt.tight_layout()
    plt.savefig('results/enhanced_v2/plots/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("✅ ENHANCED PIPELINE COMPLETE")
    logger.info("=" * 100)
    
    best_model = results_df.iloc[0]
    logger.info(f"\n🏆 Best Model: {best_model['Model']}")
    logger.info(f"   RMSE: ${best_model['rmse']:.2f}")
    logger.info(f"   MAPE: {best_model['mape']:.2f}%")
    logger.info(f"   R²:   {best_model['r2']:.4f}")
    
    logger.info("\n📁 Output Files:")
    logger.info("  • results/enhanced_v2/feature_dataset.csv")
    logger.info("  • results/enhanced_v2/model_comparison.csv")
    logger.info("  • results/enhanced_v2/plots/model_comparison.png")
    logger.info("  • results/enhanced_v2/case_studies/*.png & *.json")
    
    print("\n🎉 Enhanced training complete!")


if __name__ == "__main__":
    main()

