"""
Multi-Method Sentiment Analysis
Uses TextBlob, Vader, and FinBERT to analyze news sentiment
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import feedparser
import urllib.parse


class MultiMethodSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        print("Loading FinBERT model...")
        self.finbert_tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.finbert_model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        self.finbert_model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.finbert_model.to(self.device)
        print(f"✓ Using device: {self.device}")
    
    def textblob_sentiment(self, text):
        try:
            return TextBlob(text).sentiment.polarity
        except:
            return 0.0
    
    def vader_sentiment(self, text):
        try:
            return self.vader.polarity_scores(text)['compound']
        except:
            return 0.0
    
    def finbert_sentiment(self, text):
        try:
            inputs = self.finbert_tokenizer(text, return_tensors="pt", 
                                           truncation=True, max_length=512, 
                                           padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            sentiment = probs[0][0].item() - probs[0][2].item()
            return sentiment
        except:
            return 0.0
    
    def analyze_batch(self, texts):
        results = []
        for text in tqdm(texts, desc="Analyzing sentiment"):
            results.append({
                'textblob': self.textblob_sentiment(text),
                'vader': self.vader_sentiment(text),
                'finbert': self.finbert_sentiment(text)
            })
        return pd.DataFrame(results)


def fetch_google_news(query, start_date, end_date, max_items=500):
    url = f'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en'
    
    print(f"Fetching news from Google RSS...")
    feed = feedparser.parse(url)
    
    articles = []
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    for entry in feed.entries[:max_items]:
        try:
            pub_date = pd.to_datetime(entry.published)
            if start_dt <= pub_date <= end_dt:
                articles.append({
                    'date': pub_date.date(),
                    'title': entry.title,
                    'summary': entry.get('summary', entry.title)
                })
        except:
            continue
    
    print(f"✓ Fetched {len(articles)} articles")
    return pd.DataFrame(articles)


def compute_multi_method_sentiment(query, start_date, end_date, max_items=500):
    encoded_query = urllib.parse.quote_plus(query)
    articles_df = fetch_google_news(encoded_query, start_date, end_date, max_items)
    
    if articles_df.empty:
        print("⚠ No articles found")
        return pd.DataFrame()
    
    analyzer = MultiMethodSentimentAnalyzer()
    
    articles_df['text'] = articles_df['title'] + ' ' + articles_df['summary']
    
    sentiment_df = analyzer.analyze_batch(articles_df['text'].tolist())
    articles_df = pd.concat([articles_df, sentiment_df], axis=1)
    
    daily_sentiment = articles_df.groupby('date').agg({
        'textblob': 'mean',
        'vader': 'mean',
        'finbert': 'mean',
        'title': 'count'
    }).rename(columns={'title': 'article_count'})
    
    daily_sentiment.reset_index(inplace=True)
    
    return daily_sentiment

