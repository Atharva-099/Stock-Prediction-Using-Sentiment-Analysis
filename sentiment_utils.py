# sentiment_utils.py
import feedparser
import requests
import pandas as pd
from tqdm import tqdm
import urllib.parse

try:
    from newspaper import Article
except Exception:
    Article = None
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import datetime
import time

# Model to use: a finance-tuned FinBERT. This is compatible with the Hugging Face pipeline.
FINBERT_MODEL = "yiyanghkust/finbert-tone"  # good, finance-specific; alternative: "ProsusAI/finbert"
BATCH_SIZE = 8  # small for CPU

def fetch_google_news_rss(query, from_date=None, to_date=None, max_items=200, when_days=7):
    """
    Fetch Google News RSS items for a query.
    - query: string e.g. "Tesla" or "Tesla stock"
    - from_date/to_date: strings 'YYYY-MM-DD' or None (then uses when_days)
    Returns dataframe with columns ['date', 'title', 'link', 'summary'].
    """
    # URL encode the query to handle spaces and special characters
    q = urllib.parse.quote_plus(query)

    if from_date and to_date:
        # Google News RSS doesn't accept date range in URL reliably; we fetch recent and filter.
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    else:
        url = f"https://news.google.com/rss/search?q={q}+when:{when_days}d&hl=en-US&gl=US&ceid=US:en"

    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries[:max_items]:
        pub_date = None
        try:
            pub_date = entry.published.split("T")[0]
        except Exception:
            try:
                pub_date = datetime.datetime.fromtimestamp(time.mktime(entry.published_parsed)).strftime("%Y-%m-%d")
            except Exception:
                pub_date = None
        items.append({
            "date": pub_date,
            "title": entry.get("title"),
            "link": entry.get("link"),
            "summary": entry.get("summary", "")
        })
    df = pd.DataFrame(items)
    if from_date and to_date and not df.empty:
        df['date'] = pd.to_datetime(df['date']).dt.date
        mask = (df['date'] >= pd.to_datetime(from_date).date()) & (df['date'] <= pd.to_datetime(to_date).date())
        df = df.loc[mask]
    return df

def extract_article_text(url):
    """
    Try to download and parse article text using newspaper3k if available.
    Fallback: return empty string if parsing fails or newspaper is unavailable.
    """
    if Article is None:
        return ""
    try:
        a = Article(url)
        a.download()
        a.parse()
        text = a.text
        return text
    except Exception:
        return ""

# Initialize FinBERT pipeline once (CPU)
_tokenizer = None
_model = None
_pipeline = None

def _init_pipeline():
    global _tokenizer, _model, _pipeline
    if _pipeline is None:
        _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        # Use pipeline with device=-1 for CPU
        _pipeline = pipeline("sentiment-analysis", model=_model, tokenizer=_tokenizer, device=-1)
    return _pipeline

def finbert_batch_sentiment(texts, batch_size=BATCH_SIZE):
    """
    Run FinBERT on a list of texts (strings) and return list of dicts:
    [{'label': 'positive'|'negative'|'neutral', 'score': float}, ...]
    Numeric mapping used later: positive -> +score, negative -> -score, neutral -> 0.0
    """
    pipe = _init_pipeline()
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        out = pipe(batch)
        results.extend(out)
    return results

def compute_daily_finbert_sentiment_from_feed(query, start_date=None, end_date=None, max_items=500, fetch_article_text=True):
    """
    High-level helper:
    1) Fetch Google News RSS for query
    2) Optionally extract full article text for each link
    3) Run FinBERT and aggregate daily mean sentiment and counts
    Returns dataframe with ['date', 'finbert_mean', 'article_count']
    """
    df = fetch_google_news_rss(query, from_date=start_date, to_date=end_date, max_items=max_items)
    if df.empty:
        return pd.DataFrame(columns=["date", "finbert_mean", "article_count"])

    # Prepare text corpus: prefer article text, else title+summary
    texts = []
    dates = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing articles"):
        date = row.get("date")
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        elif pd.isna(date):
            date = None
        link = row.get("link", "")
        title = row.get("title", "") or ""
        summary = row.get("summary", "") or ""
        article_text = ""
        if fetch_article_text and link:
            article_text = extract_article_text(link)
        text = article_text if article_text else f"{title}. {summary}"
        if text.strip():
            texts.append(text)
            dates.append(date)

    if not texts:
        return pd.DataFrame(columns=["date", "finbert_mean", "article_count"])

    # Run FinBERT
    preds = finbert_batch_sentiment(texts)

    # Map labels to numeric sentiment
    numeric = []
    for r in preds:
        label = r.get("label", "").lower()
        score = float(r.get("score", 0.0))
        if "pos" in label:
            numeric.append(score)
        elif "neg" in label:
            numeric.append(-score)
        else:
            numeric.append(0.0)

    out_df = pd.DataFrame({"date": dates, "score": numeric})
    out_df['date'] = pd.to_datetime(out_df['date']).dt.date
    agg = out_df.groupby("date")["score"].agg(["mean", "count"]).reset_index()
    agg.columns = ["date", "finbert_mean", "article_count"]
    return agg
