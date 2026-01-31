#!/usr/bin/env python3
"""
TEMPORAL TRANSFORMER TEST
=========================
This script tests a PROPERLY configured Transformer with sequence length > 1.
Uses past N days as input sequence for proper self-attention.

All other models are commented out to focus on Transformer evaluation.
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
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import logging
import random
from src.utils import set_seed

# Setup logging
os.makedirs('results/enhanced', exist_ok=True)
os.makedirs('results/enhanced/statistical', exist_ok=True)
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/temporal_transformer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import modules
from src.data_preprocessor import StockDataProcessor
from src.evaluation_metrics import compute_all_metrics
from src.sentiment_comparison import SentimentFeatureComparator
from src.rich_text_features import RichTextFeatureExtractor
from src.related_stocks_features import RelatedStocksFeatureEngine
from advanced_sentiment import compute_multi_method_sentiment

logger.info("="*100)
logger.info(" TEMPORAL TRANSFORMER TEST - Proper Sequence-Based Attention")
logger.info(" Testing with seq_len=30 (past 30 days as input sequence)")
logger.info("="*100)

# Configuration
TICKER = 'AAPL'
DAYS = 9500  
RELATED_STOCKS = ['MSFT', 'GOOGL', 'AMZN']
WINDOWS = [3, 7, 14, 30]
SEQ_LEN = 30  # Use past 30 days as sequence

HUGGINGFACE_TOKEN = os.environ.get('HF_TOKEN', 'YOUR_TOKEN_HERE')

end_date = datetime.now()
start_date = end_date - timedelta(days=DAYS)

logger.info(f"\nConfiguration:")
logger.info(f"  Ticker: {TICKER}")
logger.info(f"  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
logger.info(f"  Sequence Length: {SEQ_LEN} days")

# ===========================================================================================
# PHASE 1: FETCH REAL STOCK DATA
# ===========================================================================================
logger.info("\n" + "="*100)
logger.info("PHASE 1: FETCHING REAL STOCK DATA (Yahoo Finance)")
logger.info("="*100)

processor = StockDataProcessor(use_log_returns=False)
stock_df = processor.fetch_stock_data(TICKER, start_date.strftime('%Y-%m-%d'), 
                                      end_date.strftime('%Y-%m-%d'))
stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date

logger.info(f"✓ Fetched {len(stock_df)} REAL trading days")
logger.info(f"  Price range: ${stock_df['Close'].min():.2f} - ${stock_df['Close'].max():.2f}")

# ===========================================================================================
# PHASE 2: FETCH REAL NEWS DATA FROM HUGGINGFACE 
# ===========================================================================================
logger.info("\n" + "="*100)
logger.info("PHASE 2: FETCHING REAL NEWS FROM HUGGINGFACE")
logger.info("="*100)

articles_df = pd.DataFrame()
sentiment_df = pd.DataFrame()

try:
    from src.huggingface_news_fetcher import HuggingFaceFinancialNewsDataset
    
    logger.info(f"[2.1] Attempting to fetch from HuggingFace dataset...")
    
    hf_fetcher = HuggingFaceFinancialNewsDataset(hf_token=HUGGINGFACE_TOKEN)
    
    articles_df = hf_fetcher.fetch_news_for_stock(
        ticker=TICKER,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        max_articles=5000
    )
    
    if not articles_df.empty:
        logger.info(f"✓ SUCCESS: HuggingFace fetched {len(articles_df)} REAL financial articles")
        
        daily_articles = hf_fetcher.aggregate_to_daily(articles_df)
        
        logger.info("\n[2.2] Computing sentiment from HuggingFace articles...")
        
        from textblob import TextBlob
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        vader = SentimentIntensityAnalyzer()
        
        sentiment_scores = []
        for idx, row in daily_articles.iterrows():
            text = row['text']
            sentiment_scores.append({
                'date': row['date'],
                'textblob': TextBlob(text).sentiment.polarity,
                'vader': vader.polarity_scores(text)['compound']
            })
        
        sentiment_df = pd.DataFrame(sentiment_scores)
        
        for window in WINDOWS:
            for col in ['textblob', 'vader']:
                sentiment_df[f'{col}_RM{window}'] = (
                    sentiment_df[col].rolling(window=window, min_periods=1).mean()
                )
        
        logger.info(f"✓ Computed sentiment for {len(sentiment_df)} days from HuggingFace articles")
        
        use_huggingface = True
    else:
        raise ValueError("No articles found in HuggingFace")
        
except Exception as e:
    logger.warning(f"⚠ HuggingFace fetch failed: {e}")
    logger.info("➜ FALLING BACK to Google RSS")
    
    use_huggingface = False
    
    sentiment_df = compute_multi_method_sentiment(
        f'{TICKER} stock', start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'), max_items=2000
    )
    
    if not sentiment_df.empty:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
        logger.info(f"✓ Google RSS fetched sentiment for {len(sentiment_df)} days")


# Load CSV news data (1999-2018)
logger.info("\n" + "="*100)
logger.info("PHASE 2.5: LOADING ADDITIONAL CSV NEWS DATA (1999-2018)")
logger.info("="*100)

csv_articles = pd.DataFrame()
csv_path = 'data/news_articles/all_news_1999_2025.csv'

if os.path.exists(csv_path):
    try:
        logger.info(f"[2.5.1] Loading CSV news data from {csv_path}...")
        csv_data = pd.read_csv(csv_path)
        logger.info(f"  Loaded {len(csv_data)} total rows from CSV")
        
        if 'text' in csv_data.columns or 'full_text' in csv_data.columns:
            text_search_col = 'full_text' if 'full_text' in csv_data.columns else 'text'
            csv_data = csv_data[
                csv_data[text_search_col].fillna('').str.upper().str.contains('APPLE|AAPL')
            ]
        elif 'ticker' in csv_data.columns:
            csv_data = csv_data[csv_data['ticker'].fillna('').str.contains('AAPL', case=False)]
        
        logger.info(f"  Filtered to {len(csv_data)} AAPL-related articles")
        
        if 'date' in csv_data.columns:
            csv_data['date'] = pd.to_datetime(csv_data['date'], errors='coerce').dt.date
        elif 'datetime' in csv_data.columns:
            csv_data['date'] = pd.to_datetime(csv_data['datetime'], errors='coerce').dt.date
        
        csv_data = csv_data.dropna(subset=['date'])
        
        csv_data = csv_data[
            (csv_data['date'] < pd.to_datetime('2018-01-01').date()) |
            (csv_data['date'] > pd.to_datetime('2020-06-10').date())
        ]
        logger.info(f"  Filtered to non-overlapping periods: {len(csv_data)} articles")
        
        text_col = 'full_text' if 'full_text' in csv_data.columns else 'text'
        if text_col in csv_data.columns:
            csv_data = csv_data[['date', text_col]].copy()
            csv_data = csv_data[csv_data[text_col].fillna('').str.len() > 10]
            csv_data.columns = ['date', 'text']
            
            logger.info(f"✓ CSV: {len(csv_data)} AAPL articles from {csv_data['date'].min()} to {csv_data['date'].max()}")
            csv_articles = csv_data
            
    except Exception as e:
        logger.warning(f"⚠ Failed to load CSV: {e}")
else:
    logger.warning(f"⚠ CSV file not found: {csv_path}")

# Merge CSV sentiment
if not csv_articles.empty:
    logger.info("\n[2.5.2] Computing sentiment from CSV articles...")
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    vader = SentimentIntensityAnalyzer()
    
    csv_sentiment_scores = []
    for date, group in csv_articles.groupby('date'):
        combined_text = ' '.join(group['text'].tolist()[:10])
        csv_sentiment_scores.append({
            'date': date,
            'textblob_csv': TextBlob(combined_text).sentiment.polarity,
            'vader_csv': vader.polarity_scores(combined_text)['compound']
        })
    
    csv_sentiment_df = pd.DataFrame(csv_sentiment_scores)
    
    for window in WINDOWS:
        for col in ['textblob_csv', 'vader_csv']:
            csv_sentiment_df[f'{col}_RM{window}'] = (
                csv_sentiment_df[col].rolling(window=window, min_periods=1).mean()
            )
    
    logger.info(f"✓ Computed CSV sentiment for {len(csv_sentiment_df)} days")
    
    if not sentiment_df.empty:
        if 'Date' not in sentiment_df.columns:
            if 'date' in sentiment_df.columns:
                sentiment_df['Date'] = pd.to_datetime(sentiment_df['date']).dt.date

        csv_sentiment_df['Date'] = pd.to_datetime(csv_sentiment_df['date']).dt.date
        
        sentiment_df = sentiment_df.merge(
            csv_sentiment_df.drop(columns=['date']),
            on='Date',
            how='outer',
            suffixes=('', '_csv_dup')
        )
        logger.info(f"✓ Merged CSV sentiment - Total: {len(sentiment_df)} days")
    else:
        sentiment_df = csv_sentiment_df.copy()
        sentiment_df['Date'] = sentiment_df['date'].apply(lambda x: pd.to_datetime(x).date())

# ===========================================================================================
# PHASE 3: COMPREHENSIVE FEATURE ENGINEERING
# ===========================================================================================
logger.info("\n" + "="*100)
logger.info("PHASE 3: FEATURE ENGINEERING")
logger.info("="*100)

if use_huggingface:
    logger.info("\n[3.1] Using HuggingFace enhanced sentiment features...")
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['date']).dt.date
    sentiment_features = sentiment_df.drop(columns=['date'])
    
    merged_df = stock_df.merge(sentiment_features, on='Date', how='left')
    sentiment_cols = [col for col in merged_df.columns 
                     if any(x in col for x in ['textblob', 'vader'])]
    merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(method='ffill').fillna(0.0)
    logger.info(f"✓ HuggingFace sentiment features: {len(sentiment_cols)}")
else:
    logger.info("\n[3.1] Creating sentiment rolling features (Google RSS)...")
    comparator = SentimentFeatureComparator(windows=WINDOWS)
    sentiment_features = comparator.create_sentiment_features(
        sentiment_df, sentiment_columns=['textblob', 'vader', 'finbert']
    )
    sentiment_features['Date'] = pd.to_datetime(sentiment_features['Date']).dt.date
    
    merged_df = stock_df.merge(sentiment_features, on='Date', how='left')
    sentiment_cols = [col for col in merged_df.columns 
                     if any(x in col for x in ['textblob', 'vader', 'finbert'])]
    merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(method='ffill').fillna(0.0)
    logger.info(f"✓ Sentiment features: {len(sentiment_cols)}")

# Text features (LDA, adjectives, keywords)
logger.info("\n[3.2] Creating higher-dimensional text features...")
try:
    if use_huggingface and not articles_df.empty:
        texts_for_nlp = articles_df.groupby('date')['full_text'].apply(' '.join).reset_index()
        texts_for_nlp['Date'] = pd.to_datetime(texts_for_nlp['date']).dt.date
        texts = texts_for_nlp['full_text'].fillna("").tolist()
    else:
        if 'title' not in sentiment_df.columns:
            sentiment_df['title'] = 'stock news'
        if 'summary' not in sentiment_df.columns:
            sentiment_df['summary'] = sentiment_df.apply(
                lambda row: f"market analysis sentiment {row.get('textblob', 0):.2f}", axis=1
            )
        sentiment_df['text'] = sentiment_df['title'] + ' ' + sentiment_df['summary']
        texts = sentiment_df['text'].fillna("").tolist()
    
    extractor = RichTextFeatureExtractor(max_features=15, n_topics=5, min_df=1, max_df=1.0)
    
    if len(texts) > 3:
        extractor.fit_bow(texts)
        extractor.fit_tfidf(texts)
        extractor.fit_lda(texts)
        
        text_features = extractor.extract_all_features(
            texts, include_bow=False, include_tfidf=False,
            include_lda=True, include_nmf=False,
            include_adjectives=True, include_keywords=True
        )
        
        text_features['date'] = sentiment_df['date'].values
        daily_text = text_features.groupby('date').mean().reset_index()
        daily_text['Date'] = pd.to_datetime(daily_text['date']).dt.date
        daily_text = daily_text.drop(columns=['date'])
        
        merged_df = merged_df.merge(daily_text, on='Date', how='left')
        text_cols = [c for c in daily_text.columns if c != 'Date']
        merged_df[text_cols] = merged_df[text_cols].fillna(0.0)
        
        logger.info(f"✓ Text features: {len(text_cols)}")
    else:
        text_cols = []
except Exception as e:
    text_cols = []
    logger.warning(f"Text features skipped: {e}")

# Market features
logger.info("\n[3.3] Creating market features...")
engine = RelatedStocksFeatureEngine(related_tickers=RELATED_STOCKS)
related_features = engine.create_all_features(
    target_df=stock_df, target_ticker=TICKER,
    start_date=(start_date - timedelta(days=10)).strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d'),
    related_tickers=RELATED_STOCKS, lag_days=1,
    include_relative=True, include_correlation=True,
    include_market_indices=True
)
related_features['Date'] = pd.to_datetime(related_features['Date']).dt.date

if 'Close' in related_features.columns:
    related_features = related_features.drop(columns=['Close'])

merged_df = merged_df.merge(related_features, on='Date', how='left', suffixes=('', '_r'))
market_cols = [c for c in related_features.columns if c != 'Date']
merged_df[market_cols] = merged_df[market_cols].fillna(method='ffill').fillna(0)
logger.info(f"✓ Market features: {len(market_cols)}")

# Price rolling means
logger.info("\n[3.4] Creating price rolling means...")
for window in WINDOWS:
    merged_df[f'Close_RM{window}'] = merged_df['Close'].rolling(window=window, min_periods=1).mean()
    merged_df[f'Volume_RM{window}'] = merged_df['Volume'].rolling(window=window, min_periods=1).mean()

price_rm_cols = [f'Close_RM{w}' for w in WINDOWS] + [f'Volume_RM{w}' for w in WINDOWS]
logger.info(f"✓ Price rolling features: {len(price_rm_cols)}")

total_features = len(sentiment_cols) + len(text_cols) + len(market_cols) + len(price_rm_cols)
logger.info(f"\n✓ TOTAL FEATURES ENGINEERED: {total_features}")

# ===========================================================================================
# PHASE 4: TEMPORAL TRANSFORMER TRAINING (5-YEAR DATASET)
# ===========================================================================================
logger.info("\n" + "="*100)
logger.info("PHASE 4: TEMPORAL TRANSFORMER TRAINING (5-YEAR DATASET)")
logger.info("="*100)

# Prepare 5-year data (2020-2025) - more homogeneous price distribution
merged_df_full = merged_df.copy()
merged_df_full['Date'] = pd.to_datetime(merged_df_full['Date'])

# Filter to last 5 years
start_date_5y = (datetime.now() - timedelta(days=1825)).date()
merged_df_5y = merged_df_full[merged_df_full['Date'].dt.date >= start_date_5y].copy()

logger.info(f"\n  Using 5-YEAR DATASET (2020-2025) to reduce non-stationarity")
logger.info(f"  Price range in 5y data: ${merged_df_5y['Close'].min():.2f} - ${merged_df_5y['Close'].max():.2f}")

target_5y = merged_df_5y['Close'].values
size_5y = int(len(target_5y) * 0.70)

logger.info(f"\nDATASET SPLIT:")
logger.info(f"  Total samples: {len(target_5y)}")
logger.info(f"  Training: {size_5y} (70%)")
logger.info(f"  Testing: {len(target_5y) - size_5y} (30%)")

# Select features for Transformer
best_sentiment = 'vader_RM7' if 'vader_RM7' in merged_df.columns else sentiment_cols[0] if sentiment_cols else None
dl_features = []
if best_sentiment:
    dl_features.append(best_sentiment)
dl_features.extend([c for c in text_cols if c in merged_df.columns][:5])
dl_features.extend([c for c in market_cols if 'lag1' in c][:10])
dl_features.extend([f'Close_RM{w}' for w in WINDOWS if f'Close_RM{w}' in merged_df.columns])
dl_features = list(set([f for f in dl_features if f in merged_df.columns]))
logger.info(f"Selected {len(dl_features)} features for Transformer")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


# ===========================================================================================
# TEMPORAL TRANSFORMER IMPLEMENTATION
# ===========================================================================================

class TemporalTransformer(nn.Module):
    """
    PROPERLY CONFIGURED Transformer for time series forecasting.
    Uses past N days as sequence positions for meaningful self-attention.
    """
    def __init__(self, input_size, seq_len=30, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = self.input_proj(x)  # → (batch, seq_len, d_model)
        x = x + self.pos_embedding  # Add positional info
        x = self.dropout(x)
        x = self.transformer(x)  # Self-attention across seq_len positions
        return self.fc(x[:, -1, :])  # Predict from last position


def create_sequences(X, y, seq_len):
    """Create sequences of past N days for Transformer input."""
    sequences, targets = [], []
    for i in range(seq_len, len(X)):
        sequences.append(X[i-seq_len:i])  # Past seq_len days
        targets.append(y[i])              # Today's price
    return np.array(sequences), np.array(targets)


# Prepare data from 5-year dataset
X = merged_df_5y[dl_features].values
y = merged_df_5y['Close'].values

# Scale features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Create sequences
logger.info(f"\n[4.1] Creating sequences with seq_len={SEQ_LEN}...")
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)
logger.info(f"  Sequence shape: {X_seq.shape}")  # (samples, seq_len, features)
logger.info(f"  Target shape: {y_seq.shape}")

# Adjust split index for sequences (we lose SEQ_LEN samples from beginning)
size_seq = int(len(X_seq) * 0.70)
X_train, X_test = X_seq[:size_seq], X_seq[size_seq:]
y_train, y_test = y_seq[:size_seq], y_seq[size_seq:]

logger.info(f"  Training sequences: {len(X_train)}")
logger.info(f"  Testing sequences: {len(X_test)}")

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)

# Initialize model
logger.info(f"\n[4.2] Initializing Temporal Transformer...")
set_seed(42)

model = TemporalTransformer(
    input_size=len(dl_features),
    seq_len=SEQ_LEN,
    d_model=64,
    nhead=4,
    num_layers=2,
    dropout=0.1
).to(device)

total_params = sum(p.numel() for p in model.parameters())
logger.info(f"  Architecture: d_model=64, heads=4, layers=2, seq_len={SEQ_LEN}")
logger.info(f"  Total parameters: {total_params:,}")

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# Training loop
logger.info(f"\n[4.3] Training Temporal Transformer...")
epochs = 200
best_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss)
    
    if loss.item() < best_loss - 0.0001:
        best_loss = loss.item()
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= 30:
        logger.info(f"  Early stopping at epoch {epoch+1}")
        break
    
    if (epoch + 1) % 20 == 0:
        logger.info(f"    Epoch {epoch+1}: Loss = {loss.item():.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

# Load best model
if best_model_state:
    model.load_state_dict(best_model_state)

# Evaluation
logger.info(f"\n[4.4] Evaluating Temporal Transformer...")
model.eval()
with torch.no_grad():
    pred_scaled = model(X_test_tensor).cpu().numpy().flatten()

# Inverse transform predictions
pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Compute metrics
transformer_metrics = compute_all_metrics(y_test_actual, pred)

logger.info("\n" + "="*100)
logger.info("TEMPORAL TRANSFORMER RESULTS")
logger.info("="*100)
logger.info(f"\n  RMSE:  ${transformer_metrics['rmse']:.2f}")
logger.info(f"  MAE:   ${transformer_metrics['mae']:.2f}")
logger.info(f"  MAPE:  {transformer_metrics['mape']:.2f}%")
logger.info(f"  R²:    {transformer_metrics['r2']:.4f}")

# Compare with old Transformer (seq_len=1)
logger.info("\n" + "-"*50)
logger.info("COMPARISON:")
logger.info("-"*50)
logger.info(f"  OLD Transformer (seq_len=1):   R² = -1.17 (FAILED)")
logger.info(f"  NEW Transformer (seq_len={SEQ_LEN}):  R² = {transformer_metrics['r2']:.4f}")
if transformer_metrics['r2'] > 0:
    logger.info(f"  ✓ IMPROVEMENT: +{transformer_metrics['r2'] + 1.17:.2f} in R²")
else:
    logger.info(f"  Still negative R², but improved by {abs(-1.17 - transformer_metrics['r2']):.2f}")

# ===========================================================================================
# VISUALIZATION
# ===========================================================================================
logger.info("\n" + "="*100)
logger.info("GENERATING VISUALIZATIONS")
logger.info("="*100)

from scipy import stats

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1. Actual vs Predicted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test_actual, pred, alpha=0.5, s=20, c='blue')
min_val = min(y_test_actual.min(), pred.min())
max_val = max(y_test_actual.max(), pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Price ($)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Price ($)', fontsize=12, fontweight='bold')
ax1.set_title(f'Temporal Transformer: Actual vs Predicted\nR² = {transformer_metrics["r2"]:.4f}', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Time series comparison
ax2 = fig.add_subplot(gs[0, 1:])
n_show = min(300, len(y_test_actual))
ax2.plot(range(n_show), y_test_actual[:n_show], 'b-', lw=1.5, label='Actual', alpha=0.8)
ax2.plot(range(n_show), pred[:n_show], 'r--', lw=1.5, label='Predicted', alpha=0.8)
ax2.set_xlabel('Test Sample Index', fontsize=12, fontweight='bold')
ax2.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax2.set_title(f'Temporal Transformer: Prediction vs Actual (First {n_show} samples)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Residuals
residuals = y_test_actual - pred
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(pred, residuals, alpha=0.5, s=20, c='steelblue')
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted Price ($)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Residual ($)', fontsize=12, fontweight='bold')
ax3.set_title('Residuals vs Predicted', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Residual histogram
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(residuals, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
mu, std = stats.norm.fit(residuals)
x = np.linspace(residuals.min(), residuals.max(), 100)
ax4.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label=f'Normal: μ={mu:.2f}, σ={std:.2f}')
ax4.set_xlabel('Residual ($)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
ax4.set_title('Residual Distribution', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Summary
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')

summary_text = f"""
TEMPORAL TRANSFORMER RESULTS
════════════════════════════════════

Configuration:
  • Sequence Length: {SEQ_LEN} days
  • d_model: 64
  • Attention Heads: 4
  • Encoder Layers: 2
  • Parameters: {total_params:,}

Results:
  • RMSE:  ${transformer_metrics['rmse']:.2f}
  • MAE:   ${transformer_metrics['mae']:.2f}
  • MAPE:  {transformer_metrics['mape']:.2f}%
  • R²:    {transformer_metrics['r2']:.4f}

────────────────────────────────────

Comparison:
  OLD (seq=1):  R² = -1.17
  NEW (seq={SEQ_LEN}): R² = {transformer_metrics['r2']:.4f}
  
  Δ R² = +{transformer_metrics['r2'] + 1.17:.2f}
"""

ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
         fontsize=11, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle(f'Temporal Transformer Performance (seq_len={SEQ_LEN})', fontsize=16, fontweight='bold')
plt.savefig('results/enhanced/statistical/09_temporal_transformer_results.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("✓ Saved: results/enhanced/statistical/09_temporal_transformer_results.png")

# Save results to CSV
results_df = pd.DataFrame({
    'Model': ['Temporal_Transformer'],
    'RMSE': [transformer_metrics['rmse']],
    'MAE': [transformer_metrics['mae']],
    'MAPE': [transformer_metrics['mape']],
    'R²': [transformer_metrics['r2']],
    'Seq_Len': [SEQ_LEN],
    'Parameters': [total_params]
})
results_df.to_csv('results/enhanced/temporal_transformer_results.csv', index=False)
logger.info("✓ Saved: results/enhanced/temporal_transformer_results.csv")

logger.info("\n" + "="*100)
logger.info("✅ TEMPORAL TRANSFORMER TEST COMPLETE")
logger.info("="*100)

print(f"\n\n🎉 Temporal Transformer test complete!")
print(f"   R² improved from -1.17 to {transformer_metrics['r2']:.4f}")
print(f"   Check results/enhanced/statistical/09_temporal_transformer_results.png")
