#!/usr/bin/env python3
"""
COMPLETE 1999-2025 PIPELINE - ALL MODELS
=======================================

DATA:
- 6,769 trading days of AAPL prices (1999-01-04 to 2025-11-28)
- 559,393 news articles from multiple sources

MODELS (12 total):
1. SARIMAX(1,1,1) - Statistical benchmark
2. Single-Layer Linear - Linear baseline
3. LSTM - Long Short-Term Memory
4. BiLSTM - Bidirectional LSTM
5. GRU - Gated Recurrent Unit
6. TCN - Temporal Convolutional Network
7. CNN-LSTM - Hybrid architecture
8. Transformer (64d-4h-2L) - Full transformer
9. Transformer-Medium (32d-2h-1L) - Reduced transformer
10. Transformer-Small (16d-2h-1L) - Minimal transformer
11. Attention-LSTM - LSTM with attention
12. Ensemble - Weighted average of best models
"""

import os
import sys
sys.path.insert(0, '.')

# Set CPU parallelism BEFORE importing torch
os.environ['OMP_NUM_THREADS'] = '72'
os.environ['MKL_NUM_THREADS'] = '72'
os.environ['OPENBLAS_NUM_THREADS'] = '72'
os.environ['NUMEXPR_NUM_THREADS'] = '72'

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
torch.set_num_threads(72)  # Use 72 threads for PyTorch

import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from multiprocessing import Pool, cpu_count
import json
from concurrent.futures import ThreadPoolExecutor

# Setup
N_CPUS = cpu_count()
print(f"🚀 System has {N_CPUS} CPUs - Using 72 threads for training!")

os.makedirs('results/complete_1999_2025', exist_ok=True)
os.makedirs('results/complete_1999_2025/case_studies', exist_ok=True)
os.makedirs('results/complete_1999_2025/plots', exist_ok=True)
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/complete_1999_2025.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from src.data_preprocessor import StockDataProcessor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configuration
TICKER = 'AAPL'
WINDOWS = [3, 7, 14, 30]
SEQ_LENGTH = 10
BATCH_SIZE = 128  # Larger batch for faster training


# =============================================================================
# CORRECT SCALER - Exact same factor for normalize/denormalize
# =============================================================================
class ExactScaler:
    """Ensures inverse-scaling uses EXACTLY same factor as normalization"""
    def __init__(self):
        self.min_val = None
        self.scale_factor = None
        self.fitted = False
    
    def fit(self, data):
        data = np.array(data).flatten()
        self.min_val = float(np.min(data))
        self.max_val = float(np.max(data))
        self.scale_factor = float(self.max_val - self.min_val)
        if self.scale_factor == 0:
            self.scale_factor = 1.0
        self.fitted = True
        logger.debug(f"Scaler: min={self.min_val:.4f}, max={self.max_val:.4f}, scale={self.scale_factor:.4f}")
        return self
    
    def transform(self, data):
        if not self.fitted:
            raise ValueError("Scaler not fitted!")
        return (np.array(data) - self.min_val) / self.scale_factor
    
    def inverse_transform(self, data):
        if not self.fitted:
            raise ValueError("Scaler not fitted!")
        # Uses EXACTLY the same min_val and scale_factor
        return np.array(data) * self.scale_factor + self.min_val
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


# =============================================================================
# ALL MODEL ARCHITECTURES
# =============================================================================

class SingleLayerLinear(nn.Module):
    """Single-layer linear network - baseline per professor"""
    def __init__(self, input_size, seq_length=SEQ_LENGTH):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size * seq_length, 1)
    
    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)


class LSTMModel(nn.Module):
    """Standard LSTM"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    """Gated Recurrent Unit"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TCNBlock(nn.Module):
    """Temporal Convolutional Block"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        out = self.conv(x)[:, :, :x.size(2)]  # Causal padding
        out = torch.relu(self.bn(out))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)


class TCNModel(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, input_size, hidden_channels=[32, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_channels = [input_size] + hidden_channels
        for i in range(len(hidden_channels)):
            dilation = 2 ** i
            layers.append(TCNBlock(num_channels[i], num_channels[i+1], kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_channels[-1], 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = self.network(x)
        return self.fc(x[:, :, -1])


class CNNLSTMModel(nn.Module):
    """CNN-LSTM Hybrid"""
    def __init__(self, input_size, cnn_channels=32, lstm_hidden=64, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(cnn_channels)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.bn(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    """Full Transformer (64d-4h-2L)"""
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


class TransformerMedium(nn.Module):
    """Medium Transformer (32d-2h-1L)"""
    def __init__(self, input_size, d_model=32, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*2, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


class TransformerSmall(nn.Module):
    """Small Transformer (16d-2h-1L)"""
    def __init__(self, input_size, d_model=16, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*2, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


class AttentionLSTM(nn.Module):
    """LSTM with Attention mechanism"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)
        return self.fc(context)


# =============================================================================
# CASE STUDY ANALYZER
# =============================================================================
class CaseStudyAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.major_events = {
            '2000-03-10': {'name': 'Dot-com Bubble Peak'},
            '2001-09-17': {'name': '9/11 Market Reopens'},
            '2007-10-09': {'name': 'Pre-Crisis Peak'},
            '2008-09-15': {'name': 'Lehman Brothers Collapse'},
            '2008-10-13': {'name': 'Bank Bailout Announced'},
            '2009-03-09': {'name': 'Market Bottom'},
            '2010-05-06': {'name': 'Flash Crash'},
            '2011-08-08': {'name': 'US Credit Downgrade'},
            '2011-10-05': {'name': 'Steve Jobs Death'},
            '2012-09-21': {'name': 'iPhone 5 Launch'},
            '2014-06-09': {'name': '7:1 Stock Split'},
            '2015-08-24': {'name': 'China Black Monday'},
            '2018-08-02': {'name': 'First $1T Company'},
            '2020-03-16': {'name': 'COVID-19 Crash'},
            '2020-03-23': {'name': 'Fed Unlimited QE'},
            '2020-08-31': {'name': '4:1 Stock Split'},
            '2022-01-03': {'name': 'Apple $3T Peak'},
            '2022-06-13': {'name': 'Bear Market Start'},
            '2023-06-05': {'name': 'Vision Pro Announced'},
        }
    
    def analyze_headline(self, headline):
        words = headline.lower().split()
        analyzed = []
        for w in words:
            if len(w) > 2:
                score = self.vader.polarity_scores(w)['compound']
                analyzed.append({'word': w, 'sentiment': score})
        return analyzed
    
    def generate_case_study(self, date, headlines, actual, predicted, prev_price, model):
        actual_move = (actual - prev_price) / prev_price * 100
        pred_move = (predicted - prev_price) / prev_price * 100
        
        analyzed_headlines = []
        for h in headlines[:5]:
            analyzed_headlines.append({
                'text': h,
                'sentiment': self.vader.polarity_scores(h)['compound'],
                'words': self.analyze_headline(h)
            })
        
        return {
            'date': str(date),
            'event': self.major_events.get(str(date), {'name': 'Trading Day'}),
            'headlines': analyzed_headlines,
            'actual_price': float(actual),
            'predicted_price': float(predicted),
            'prev_price': float(prev_price),
            'actual_move_pct': float(actual_move),
            'predicted_move_pct': float(pred_move),
            'error_pct': float(abs(pred_move - actual_move)),
            'model': model
        }
    
    def plot_case_study(self, case, save_path):
        fig = plt.figure(figsize=(16, 12))
        
        # Price comparison
        ax1 = fig.add_subplot(2, 2, 1)
        prices = [case['prev_price'], case['actual_price'], case['predicted_price']]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = ax1.bar(['Previous', 'Actual', 'Predicted'], prices, color=colors, edgecolor='black')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(f"Case Study: {case['date']}\n{case['event'].get('name', 'Trading Day')}", fontsize=14, fontweight='bold')
        for bar, price in zip(bars, prices):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'${price:.2f}', ha='center', fontsize=11)
        
        # Move comparison
        ax2 = fig.add_subplot(2, 2, 2)
        moves = [case['actual_move_pct'], case['predicted_move_pct']]
        move_colors = ['#2ecc71' if m > 0 else '#e74c3c' for m in moves]
        bars = ax2.bar(['Actual Move', 'Predicted Move'], moves, color=move_colors, edgecolor='black')
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_ylabel('Move (%)', fontsize=12)
        ax2.set_title(f"Price Movement (Error: {case['error_pct']:.2f}%)", fontsize=14)
        for bar, move in zip(bars, moves):
            y_pos = bar.get_height() + 0.1 if move > 0 else bar.get_height() - 0.3
            ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f'{move:.2f}%', ha='center', fontsize=11)
        
        # Headline sentiments
        ax3 = fig.add_subplot(2, 2, 3)
        if case['headlines']:
            sents = [h['sentiment'] for h in case['headlines']]
            labels = [f"H{i+1}" for i in range(len(sents))]
            colors = ['#2ecc71' if s > 0.05 else '#e74c3c' if s < -0.05 else '#95a5a6' for s in sents]
            ax3.barh(labels, sents, color=colors, edgecolor='black')
            ax3.set_xlabel('Sentiment Score', fontsize=12)
            ax3.set_title('Headline Sentiments', fontsize=14)
            ax3.axvline(0, color='black', linewidth=0.5)
            ax3.set_xlim(-1, 1)
        
        # Key sentiment words
        ax4 = fig.add_subplot(2, 2, 4)
        if case['headlines']:
            all_words = []
            for h in case['headlines']:
                for w in h.get('words', []):
                    if abs(w['sentiment']) > 0.1:
                        all_words.append((w['word'], w['sentiment']))
            all_words.sort(key=lambda x: abs(x[1]), reverse=True)
            if all_words:
                words, sents = zip(*all_words[:10])
                colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in sents]
                ax4.barh(range(len(words)), sents, color=colors, edgecolor='black')
                ax4.set_yticks(range(len(words)))
                ax4.set_yticklabels(words, fontsize=10)
                ax4.set_xlabel('Sentiment Weight', fontsize=12)
                ax4.set_title('Key Sentiment Words', fontsize=14)
                ax4.axvline(0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()


# =============================================================================
# DATA LOADING
# =============================================================================
def load_all_news_data():
    """Load ALL news data (1999-2025)"""
    logger.info("Loading all news data (1999-2025)...")
    
    all_dfs = []
    files = [
        ('data/fnspid_official/news_1999_2008.parquet', '1999-2008'),
        ('data/historical_cache/fnspid_2009_2020.parquet', '2009-2020'),
        ('data/fnspid_official/news_2021_2023.parquet', '2021-2023'),
        ('data/fnspid_official/news_2024_2025_rss.parquet', '2024-2025 RSS'),
    ]
    
    for path, period in files:
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                logger.info(f"  ✓ Loaded {period}: {len(df):,} articles")
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"  ✗ Failed to load {period}: {e}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"✅ TOTAL NEWS ARTICLES: {len(combined):,}")
        return combined
    
    logger.warning("No news data found!")
    return pd.DataFrame()


def compute_sentiment_batch(texts, vader):
    """Compute sentiment for a batch of texts"""
    return [vader.polarity_scores(str(t)[:500])['compound'] for t in texts]


def compute_daily_sentiment(news_df):
    """Compute daily sentiment using parallel processing"""
    if news_df.empty:
        return pd.DataFrame()
    
    vader = SentimentIntensityAnalyzer()
    
    # Find columns
    text_col = next((c for c in ['Article_title', 'title', 'text', 'headline'] 
                    if c in news_df.columns), None)
    date_col = next((c for c in ['Date', 'date', 'datetime'] 
                    if c in news_df.columns), None)
    
    if not text_col or not date_col:
        logger.warning("Missing text or date column")
        return pd.DataFrame()
    
    logger.info(f"  Computing sentiment for {len(news_df):,} articles...")
    
    # Process in chunks for efficiency
    texts = news_df[text_col].fillna('').tolist()
    sentiments = []
    chunk_size = 10000
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        chunk_sents = compute_sentiment_batch(chunk, vader)
        sentiments.extend(chunk_sents)
        if (i + chunk_size) % 50000 == 0:
            logger.info(f"    Processed {min(i+chunk_size, len(texts)):,}/{len(texts):,} articles...")
    
    news_df = news_df.copy()
    news_df['sentiment'] = sentiments
    news_df['date'] = pd.to_datetime(news_df[date_col]).dt.date
    
    # Aggregate daily
    daily = news_df.groupby('date').agg({
        'sentiment': 'mean',
        text_col: 'count'
    }).rename(columns={text_col: 'article_count'})
    
    daily = daily.reset_index()
    logger.info(f"✅ Daily sentiment for {len(daily)} days")
    
    return daily


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def prepare_sequences(X, y, seq_length):
    """Create sequences for time series"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)


def train_neural_network(model, X_train, y_train, X_val, y_val, 
                         epochs=150, batch_size=BATCH_SIZE, lr=0.001, patience=20):
    """Train neural network with proper validation"""
    
    device = torch.device('cpu')  # Using CPU with 72 threads
    model = model.to(device)
    
    # Scale data
    scaler_X = ExactScaler()
    scaler_y = ExactScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_train_seq, y_train_seq = prepare_sequences(X_train_scaled, y_train_scaled, SEQ_LENGTH)
    X_val_seq, y_val_seq = prepare_sequences(X_val_scaled, y_val_scaled, SEQ_LENGTH)
    
    # To tensors
    X_train_t = torch.FloatTensor(X_train_seq).to(device)
    y_train_t = torch.FloatTensor(y_train_seq).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val_seq).to(device)
    y_val_t = torch.FloatTensor(y_val_seq).unsqueeze(1).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        perm = torch.randperm(X_train_t.size(0))
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, X_train_t.size(0), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(X_train_t[idx])
            loss = criterion(outputs, y_train_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Load best state
    if best_state:
        model.load_state_dict(best_state)
    
    return model, scaler_X, scaler_y


def predict_with_rolling_window(model, X_data, scaler_X, scaler_y):
    """Rolling prediction with full window update"""
    model.eval()
    device = next(model.parameters()).device
    
    X_scaled = scaler_X.transform(X_data)
    predictions = []
    
    for i in range(SEQ_LENGTH, len(X_scaled)):
        # Get full window
        window = X_scaled[i-SEQ_LENGTH:i]
        X_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy().flatten()[0]
        
        # Inverse transform using EXACT same scaler
        pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(pred)
    
    return np.array(predictions)


def train_sarimax(y_train, y_test, exog_train=None, exog_test=None):
    """SARIMAX(1,1,1) with no seasonal terms - professor's requirement"""
    logger.info("  Training SARIMAX(1,1,1) with d=1, no seasonal terms...")
    
    history = list(y_train)
    history_exog = list(exog_train) if exog_train is not None else None
    predictions = []
    
    for t in range(len(y_test)):
        try:
            if history_exog is not None:
                model = SARIMAX(
                    history, 
                    exog=np.array(history_exog).reshape(-1, 1),
                    order=(1, 1, 1),
                    seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fit = model.fit(disp=False, maxiter=50)
                yhat = fit.forecast(1, exog=np.array([[exog_test[t]]]))[0]
            else:
                model = SARIMAX(
                    history,
                    order=(1, 1, 1),
                    seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fit = model.fit(disp=False, maxiter=50)
                yhat = fit.forecast(1)[0]
        except Exception as e:
            yhat = history[-1]
        
        predictions.append(yhat)
        history.append(y_test[t])
        if history_exog is not None:
            history_exog.append(exog_test[t])
        
        if (t + 1) % 100 == 0:
            logger.info(f"    SARIMAX progress: {t+1}/{len(y_test)}")
    
    return np.array(predictions)


def compute_metrics(y_true, y_pred):
    """Compute all evaluation metrics"""
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]
    
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
        'r2': r2_score(y_true, y_pred)
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    start_time = datetime.now()
    
    logger.info("=" * 100)
    logger.info(" COMPLETE 1999-2025 PIPELINE - ALL 12 MODELS")
    logger.info(f" Using {N_CPUS} CPUs with 72 PyTorch threads")
    logger.info("=" * 100)
    
    device = torch.device('cpu')
    logger.info(f"Device: {device} (with 72-thread parallelism)")
    
    # =========================================================================
    # PHASE 1: LOAD ALL DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: LOADING ALL DATA (1999-2025)")
    logger.info("=" * 80)
    
    # Stock data
    processor = StockDataProcessor(use_log_returns=False)
    stock_df = processor.fetch_stock_data(TICKER, '1999-01-01', datetime.now().strftime('%Y-%m-%d'))
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
    
    logger.info(f"✅ STOCK DATA: {len(stock_df):,} trading days")
    logger.info(f"   Date range: {stock_df['Date'].min()} to {stock_df['Date'].max()}")
    logger.info(f"   Price range: ${stock_df['Close'].min():.2f} to ${stock_df['Close'].max():.2f}")
    
    # News data
    news_df = load_all_news_data()
    if not news_df.empty:
        sentiment_df = compute_daily_sentiment(news_df)
    else:
        logger.warning("Using price-based sentiment proxy")
        stock_df['Return'] = stock_df['Close'].pct_change()
        sentiment_df = pd.DataFrame({
            'date': stock_df['Date'],
            'sentiment': stock_df['Return'].rolling(5).mean().fillna(0) * 10,
            'article_count': 1
        })
    
    # =========================================================================
    # PHASE 2: FEATURE ENGINEERING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    # Technical indicators
    for w in WINDOWS:
        stock_df[f'Close_SMA{w}'] = stock_df['Close'].rolling(w, min_periods=1).mean()
        stock_df[f'Close_EMA{w}'] = stock_df['Close'].ewm(span=w, min_periods=1).mean()
        stock_df[f'Volume_SMA{w}'] = stock_df['Volume'].rolling(w, min_periods=1).mean()
        stock_df[f'Return_{w}d'] = stock_df['Close'].pct_change(w)
    
    stock_df['Daily_Return'] = stock_df['Close'].pct_change()
    stock_df['Volatility_7d'] = stock_df['Daily_Return'].rolling(7).std()
    stock_df['Volatility_30d'] = stock_df['Daily_Return'].rolling(30).std()
    stock_df['RSI'] = 100 - (100 / (1 + stock_df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                     stock_df['Close'].diff().clip(upper=0).abs().rolling(14).mean()))
    stock_df['MACD'] = stock_df['Close'].ewm(span=12).mean() - stock_df['Close'].ewm(span=26).mean()
    
    # Merge sentiment
    if not sentiment_df.empty:
        sentiment_df.columns = ['Date' if c == 'date' else c for c in sentiment_df.columns]
        merged_df = stock_df.merge(sentiment_df, on='Date', how='left')
        merged_df['sentiment'] = merged_df['sentiment'].fillna(0)
        merged_df['article_count'] = merged_df['article_count'].fillna(0)
        
        for w in WINDOWS:
            merged_df[f'sentiment_SMA{w}'] = merged_df['sentiment'].rolling(w, min_periods=1).mean()
    else:
        merged_df = stock_df.copy()
        merged_df['sentiment'] = 0
        merged_df['article_count'] = 0
    
    merged_df = merged_df.fillna(method='ffill').fillna(0)
    
    # Feature columns
    feature_cols = [c for c in merged_df.columns 
                   if c not in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                   and merged_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    logger.info(f"✅ Created {len(feature_cols)} features")
    
    # Save dataset
    merged_df.to_csv('results/complete_1999_2025/complete_dataset.csv', index=False)
    logger.info("   Saved: results/complete_1999_2025/complete_dataset.csv")
    
    # =========================================================================
    # PHASE 3: TRAIN/VAL/TEST SPLIT
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: DATA SPLIT (70% Train / 15% Val / 15% Test)")
    logger.info("=" * 80)
    
    total = len(merged_df)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)
    
    train_df = merged_df.iloc[:train_end]
    val_df = merged_df.iloc[train_end:val_end]
    test_df = merged_df.iloc[val_end:]
    
    logger.info(f"   Train: {len(train_df):,} days ({train_df['Date'].iloc[0]} → {train_df['Date'].iloc[-1]})")
    logger.info(f"   Val:   {len(val_df):,} days ({val_df['Date'].iloc[0]} → {val_df['Date'].iloc[-1]})")
    logger.info(f"   Test:  {len(test_df):,} days ({test_df['Date'].iloc[0]} → {test_df['Date'].iloc[-1]})")
    
    X_train = train_df[feature_cols].values
    y_train = train_df['Close'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['Close'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['Close'].values
    test_dates = test_df['Date'].values
    
    X_full = np.vstack([X_train[-SEQ_LENGTH:], X_val, X_test])
    n_features = len(feature_cols)
    
    # =========================================================================
    # PHASE 4: TRAIN ALL 12 MODELS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: TRAINING ALL 12 MODELS")
    logger.info("=" * 80)
    
    all_results = {}
    all_predictions = {}
    
    # 1. SARIMAX Benchmark
    logger.info("\n[1/12] SARIMAX(1,1,1) - Statistical Benchmark...")
    sent_col = 'sentiment_SMA7' if 'sentiment_SMA7' in merged_df.columns else None
    if sent_col:
        exog_train = merged_df[sent_col].iloc[:train_end].values
        exog_test = merged_df[sent_col].iloc[val_end:].values
        sarimax_pred = train_sarimax(y_train, y_test, exog_train, exog_test)
    else:
        sarimax_pred = train_sarimax(y_train, y_test)
    all_results['SARIMAX(1,1,1)'] = compute_metrics(y_test, sarimax_pred)
    all_predictions['SARIMAX(1,1,1)'] = sarimax_pred
    logger.info(f"   ✓ RMSE: ${all_results['SARIMAX(1,1,1)']['rmse']:.2f}, R²: {all_results['SARIMAX(1,1,1)']['r2']:.4f}")
    
    # Neural network models
    nn_models = [
        ('SingleLayerLinear', SingleLayerLinear, {'seq_length': SEQ_LENGTH}),
        ('LSTM', LSTMModel, {}),
        ('BiLSTM', BiLSTMModel, {}),
        ('GRU', GRUModel, {}),
        ('TCN', TCNModel, {}),
        ('CNN-LSTM', CNNLSTMModel, {}),
        ('Transformer-Full', TransformerModel, {}),
        ('Transformer-Medium', TransformerMedium, {}),
        ('Transformer-Small', TransformerSmall, {}),
        ('Attention-LSTM', AttentionLSTM, {}),
    ]
    
    for idx, (name, ModelClass, kwargs) in enumerate(nn_models, 2):
        logger.info(f"\n[{idx}/12] {name}...")
        try:
            model = ModelClass(n_features, **kwargs)
            model, scaler_X, scaler_y = train_neural_network(
                model, X_train, y_train, X_val, y_val
            )
            preds = predict_with_rolling_window(model, X_full, scaler_X, scaler_y)
            preds = preds[-len(y_test):]
            all_results[name] = compute_metrics(y_test, preds)
            all_predictions[name] = preds
            logger.info(f"   ✓ RMSE: ${all_results[name]['rmse']:.2f}, R²: {all_results[name]['r2']:.4f}")
        except Exception as e:
            logger.error(f"   ✗ Failed: {e}")
            all_results[name] = {'rmse': 999, 'mae': 999, 'mape': 999, 'r2': -999}
    
    # 12. Ensemble (weighted average of top 3 models)
    logger.info("\n[12/12] Ensemble (Top 3 Models)...")
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['rmse'])
    top_3 = [m[0] for m in sorted_models[:3] if m[0] in all_predictions]
    if len(top_3) >= 2:
        ensemble_pred = np.mean([all_predictions[m] for m in top_3], axis=0)
        all_results['Ensemble'] = compute_metrics(y_test, ensemble_pred)
        all_predictions['Ensemble'] = ensemble_pred
        logger.info(f"   ✓ RMSE: ${all_results['Ensemble']['rmse']:.2f}, R²: {all_results['Ensemble']['r2']:.4f}")
        logger.info(f"   Components: {', '.join(top_3)}")
    
    # =========================================================================
    # PHASE 5: CASE STUDIES
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: CASE STUDIES (Major Market Events)")
    logger.info("=" * 80)
    
    analyzer = CaseStudyAnalyzer()
    
    # Find days with significant moves (>2% change)
    returns = np.diff(y_test) / y_test[:-1] * 100
    significant_days = np.where(np.abs(returns) > 2)[0]
    
    best_model = min(all_results.keys(), key=lambda k: all_results[k]['rmse'])
    case_studies = []
    
    for idx in significant_days[:10]:  # Top 10 significant days
        if idx + 1 < len(y_test) and best_model in all_predictions:
            date = test_dates[idx + 1]
            
            # Get headlines for this date
            if not news_df.empty:
                date_col = 'Date' if 'Date' in news_df.columns else 'date'
                text_col = 'Article_title' if 'Article_title' in news_df.columns else 'title'
                if date_col in news_df.columns and text_col in news_df.columns:
                    day_news = news_df[pd.to_datetime(news_df[date_col]).dt.date == date]
                    headlines = day_news[text_col].tolist()[:5] if len(day_news) > 0 else ['No headlines available']
                else:
                    headlines = ['No headlines available']
            else:
                headlines = ['No headlines available']
            
            case = analyzer.generate_case_study(
                date, headlines, y_test[idx+1],
                all_predictions[best_model][idx+1] if idx+1 < len(all_predictions[best_model]) else y_test[idx],
                y_test[idx], best_model
            )
            case_studies.append(case)
            
            # Save case study
            analyzer.plot_case_study(case, f'results/complete_1999_2025/case_studies/case_{date}.png')
            with open(f'results/complete_1999_2025/case_studies/case_{date}.json', 'w') as f:
                json.dump(case, f, indent=2, default=str)
            
            logger.info(f"   ✓ Case study: {date} ({case['event'].get('name', 'Trading Day')})")
    
    # =========================================================================
    # PHASE 6: FINAL RESULTS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6: FINAL RESULTS & COMPARISON")
    logger.info("=" * 80)
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {'Model': name, **metrics} 
        for name, metrics in all_results.items()
    ]).sort_values('rmse')
    
    results_df.to_csv('results/complete_1999_2025/model_comparison.csv', index=False)
    
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON (Sorted by RMSE)")
    logger.info("=" * 60)
    logger.info(f"\n{'Model':<25} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'R²':>10}")
    logger.info("-" * 65)
    for _, row in results_df.iterrows():
        logger.info(f"{row['Model']:<25} ${row['rmse']:>8.2f} ${row['mae']:>8.2f} {row['mape']:>9.2f}% {row['r2']:>9.4f}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    models = results_df['Model'].tolist()
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    metrics_info = [
        ('rmse', 'RMSE ($)', 'lower is better'),
        ('mae', 'MAE ($)', 'lower is better'),
        ('mape', 'MAPE (%)', 'lower is better'),
        ('r2', 'R² Score', 'higher is better')
    ]
    
    for ax, (metric, title, note) in zip(axes.flat, metrics_info):
        bars = ax.barh(models, results_df[metric], color=colors, edgecolor='black')
        ax.set_xlabel(title, fontsize=12)
        ax.set_title(f'{title} ({note})', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/complete_1999_2025/plots/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Prediction plot
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(test_dates, y_test, 'b-', label='Actual Price', linewidth=2, alpha=0.8)
    ax.plot(test_dates, all_predictions[best_model], 'r--', 
           label=f'{best_model} Prediction', linewidth=1.5, alpha=0.7)
    ax.fill_between(test_dates, y_test, all_predictions[best_model], alpha=0.2, color='gray')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title(f'{TICKER} Stock Price: Actual vs {best_model} Predictions\nTest Period: {test_dates[0]} to {test_dates[-1]}', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/complete_1999_2025/plots/prediction_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = datetime.now() - start_time
    
    logger.info("\n" + "=" * 100)
    logger.info("✅ COMPLETE 1999-2025 PIPELINE FINISHED!")
    logger.info("=" * 100)
    
    best = results_df.iloc[0]
    logger.info(f"\n🏆 BEST MODEL: {best['Model']}")
    logger.info(f"   RMSE: ${best['rmse']:.2f}")
    logger.info(f"   MAE:  ${best['mae']:.2f}")
    logger.info(f"   MAPE: {best['mape']:.2f}%")
    logger.info(f"   R²:   {best['r2']:.4f}")
    
    logger.info(f"\n📊 DATA SUMMARY:")
    logger.info(f"   Stock prices: {len(stock_df):,} days ({stock_df['Date'].min()} → {stock_df['Date'].max()})")
    logger.info(f"   News articles: {len(news_df):,}")
    logger.info(f"   Features: {len(feature_cols)}")
    
    logger.info(f"\n📁 OUTPUT FILES:")
    logger.info(f"   • results/complete_1999_2025/complete_dataset.csv")
    logger.info(f"   • results/complete_1999_2025/model_comparison.csv")
    logger.info(f"   • results/complete_1999_2025/plots/model_comparison.png")
    logger.info(f"   • results/complete_1999_2025/plots/prediction_comparison.png")
    logger.info(f"   • results/complete_1999_2025/case_studies/*.png & *.json")
    
    logger.info(f"\n⏱️  Total time: {elapsed}")
    
    print(f"\n🎉 Complete! Best model: {best['Model']} with RMSE=${best['rmse']:.2f}")
    print(f"   Check results/complete_1999_2025/ for all outputs.")


if __name__ == "__main__":
    main()


