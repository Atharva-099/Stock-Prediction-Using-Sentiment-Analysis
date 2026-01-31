"""
Rich Text Feature Extraction Module
Implements advanced NLP features beyond simple sentiment scores:
- Bag-of-Words (BoW) and TF-IDF
- Topic Modeling (LDA)
- Adjective frequency tracking
- Named Entity Recognition
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import logging

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RichTextFeatureExtractor:
    """
    Extract rich text features from news articles for stock prediction
    """
    
    def __init__(self, 
                 max_features: int = 100,
                 n_topics: int = 5,
                 min_df: int = 2,
                 max_df: float = 0.8):
        """
        Args:
            max_features: Maximum number of features for BoW/TF-IDF
            n_topics: Number of topics for LDA
            min_df: Minimum document frequency
            max_df: Maximum document frequency (as ratio)
        """
        self.max_features = max_features
        self.n_topics = n_topics
        self.min_df = min_df
        self.max_df = max_df
        
        # Vectorizers
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        
        # Topic models
        self.lda_model = None
        self.nmf_model = None
        
        # Lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Stop words (English + financial stopwords)
        self.stop_words = set(stopwords.words('english'))
        
        # Financial keywords to track
        self.financial_keywords = [
            'revenue', 'profit', 'loss', 'earnings', 'growth', 'decline',
            'bullish', 'bearish', 'rally', 'crash', 'volatility', 'risk',
            'investment', 'stock', 'share', 'dividend', 'acquisition', 'merger'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text: lowercase, tokenize, lemmatize
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token.isalnum() and token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_adjectives(self, text: str) -> List[str]:
        """
        Extract adjectives from text (important for sentiment)
        
        Args:
            text: Input text
            
        Returns:
            List of adjectives
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        try:
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # Extract adjectives (JJ, JJR, JJS tags)
            adjectives = [word for word, tag in pos_tags 
                         if tag in ['JJ', 'JJR', 'JJS']]
            
            return adjectives
        except Exception as e:
            logger.debug(f"Error extracting adjectives: {e}")
            return []
    
    def compute_adjective_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Compute adjective-based features
        
        Args:
            texts: List of text documents
            
        Returns:
            DataFrame with adjective features
        """
        logger.info("Computing adjective features...")
        
        features = []
        
        for text in texts:
            adjectives = self.extract_adjectives(text)
            
            # Count adjectives
            adj_count = len(adjectives)
            
            # Unique adjectives
            unique_adj = len(set(adjectives))
            
            # Adjective density (adjectives per word)
            word_count = len(text.split()) if text else 1
            adj_density = adj_count / word_count if word_count > 0 else 0
            
            # Positive/negative adjectives (simple heuristic)
            positive_adj = ['good', 'great', 'excellent', 'strong', 'positive', 
                           'high', 'better', 'best', 'bullish', 'profitable']
            negative_adj = ['bad', 'poor', 'weak', 'negative', 'low', 'worse', 
                           'worst', 'bearish', 'loss', 'declining']
            
            pos_count = sum(1 for adj in adjectives if adj in positive_adj)
            neg_count = sum(1 for adj in adjectives if adj in negative_adj)
            
            features.append({
                'adj_count': adj_count,
                'adj_unique': unique_adj,
                'adj_density': adj_density,
                'adj_positive': pos_count,
                'adj_negative': neg_count,
                'adj_sentiment': pos_count - neg_count
            })
        
        return pd.DataFrame(features)
    
    def fit_bow(self, texts: List[str]) -> 'RichTextFeatureExtractor':
        """
        Fit Bag-of-Words vectorizer
        
        Args:
            texts: List of text documents
            
        Returns:
            self
        """
        logger.info(f"Fitting BoW vectorizer (max_features={self.max_features})...")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        self.bow_vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english'
        )
        
        self.bow_vectorizer.fit(processed_texts)
        
        logger.info(f"BoW vocabulary size: {len(self.bow_vectorizer.vocabulary_)}")
        
        return self
    
    def fit_tfidf(self, texts: List[str]) -> 'RichTextFeatureExtractor':
        """
        Fit TF-IDF vectorizer
        
        Args:
            texts: List of text documents
            
        Returns:
            self
        """
        logger.info(f"Fitting TF-IDF vectorizer (max_features={self.max_features})...")
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english'
        )
        
        self.tfidf_vectorizer.fit(processed_texts)
        
        logger.info(f"TF-IDF vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        return self
    
    def fit_lda(self, texts: List[str]) -> 'RichTextFeatureExtractor':
        """
        Fit Latent Dirichlet Allocation (LDA) topic model
        
        Args:
            texts: List of text documents
            
        Returns:
            self
        """
        logger.info(f"Fitting LDA topic model (n_topics={self.n_topics})...")
        
        # First fit BoW if not already fitted
        if self.bow_vectorizer is None:
            self.fit_bow(texts)
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        bow_features = self.bow_vectorizer.transform(processed_texts)
        
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=20,
            n_jobs=-1
        )
        
        self.lda_model.fit(bow_features)
        
        logger.info("LDA model fitted successfully")
        
        return self
    
    def fit_nmf(self, texts: List[str]) -> 'RichTextFeatureExtractor':
        """
        Fit Non-negative Matrix Factorization (NMF) topic model
        
        Args:
            texts: List of text documents
            
        Returns:
            self
        """
        logger.info(f"Fitting NMF topic model (n_topics={self.n_topics})...")
        
        # First fit TF-IDF if not already fitted
        if self.tfidf_vectorizer is None:
            self.fit_tfidf(texts)
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        tfidf_features = self.tfidf_vectorizer.transform(processed_texts)
        
        self.nmf_model = NMF(
            n_components=self.n_topics,
            random_state=42,
            max_iter=200
        )
        
        self.nmf_model.fit(tfidf_features)
        
        logger.info("NMF model fitted successfully")
        
        return self
    
    def transform_bow(self, texts: List[str]) -> np.ndarray:
        """Transform texts to BoW features"""
        if self.bow_vectorizer is None:
            raise ValueError("BoW vectorizer not fitted. Call fit_bow first.")
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.bow_vectorizer.transform(processed_texts).toarray()
    
    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF features"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.tfidf_vectorizer.transform(processed_texts).toarray()
    
    def transform_lda(self, texts: List[str]) -> np.ndarray:
        """Transform texts to LDA topic distributions"""
        if self.lda_model is None:
            raise ValueError("LDA model not fitted. Call fit_lda first.")
        
        bow_features = self.transform_bow(texts)
        return self.lda_model.transform(bow_features)
    
    def transform_nmf(self, texts: List[str]) -> np.ndarray:
        """Transform texts to NMF topic distributions"""
        if self.nmf_model is None:
            raise ValueError("NMF model not fitted. Call fit_nmf first.")
        
        tfidf_features = self.transform_tfidf(texts)
        return self.nmf_model.transform(tfidf_features)
    
    def extract_keyword_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract financial keyword presence features
        
        Args:
            texts: List of text documents
            
        Returns:
            DataFrame with keyword features
        """
        logger.info("Extracting financial keyword features...")
        
        features = []
        
        for text in texts:
            text_lower = text.lower() if isinstance(text, str) else ""
            
            keyword_counts = {f'kw_{kw}': text_lower.count(kw) 
                            for kw in self.financial_keywords}
            
            features.append(keyword_counts)
        
        return pd.DataFrame(features)
    
    def extract_all_features(self, 
                            texts: List[str],
                            include_bow: bool = True,
                            include_tfidf: bool = True,
                            include_lda: bool = True,
                            include_nmf: bool = False,
                            include_adjectives: bool = True,
                            include_keywords: bool = True) -> pd.DataFrame:
        """
        Extract all text features
        
        Args:
            texts: List of text documents
            include_bow: Include BoW features
            include_tfidf: Include TF-IDF features
            include_lda: Include LDA topics
            include_nmf: Include NMF topics
            include_adjectives: Include adjective features
            include_keywords: Include keyword features
            
        Returns:
            DataFrame with all features
        """
        logger.info("Extracting all text features...")
        
        all_features = []
        
        # Adjective features
        if include_adjectives:
            adj_features = self.compute_adjective_features(texts)
            all_features.append(adj_features)
        
        # Keyword features
        if include_keywords:
            kw_features = self.extract_keyword_features(texts)
            all_features.append(kw_features)
        
        # BoW features (reduced to top N)
        if include_bow and self.bow_vectorizer is not None:
            bow_features = self.transform_bow(texts)
            bow_df = pd.DataFrame(
                bow_features,
                columns=[f'bow_{i}' for i in range(bow_features.shape[1])]
            )
            all_features.append(bow_df)
        
        # TF-IDF features (reduced to top N)
        if include_tfidf and self.tfidf_vectorizer is not None:
            tfidf_features = self.transform_tfidf(texts)
            tfidf_df = pd.DataFrame(
                tfidf_features,
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )
            all_features.append(tfidf_df)
        
        # LDA topics
        if include_lda and self.lda_model is not None:
            lda_features = self.transform_lda(texts)
            lda_df = pd.DataFrame(
                lda_features,
                columns=[f'lda_topic_{i}' for i in range(lda_features.shape[1])]
            )
            all_features.append(lda_df)
        
        # NMF topics
        if include_nmf and self.nmf_model is not None:
            nmf_features = self.transform_nmf(texts)
            nmf_df = pd.DataFrame(
                nmf_features,
                columns=[f'nmf_topic_{i}' for i in range(nmf_features.shape[1])]
            )
            all_features.append(nmf_df)
        
        # Concatenate all features
        if all_features:
            combined_df = pd.concat(all_features, axis=1)
            logger.info(f"Extracted {combined_df.shape[1]} text features")
            return combined_df
        else:
            logger.warning("No features extracted")
            return pd.DataFrame()
    
    def get_top_words_per_topic(self, n_words: int = 10) -> Dict[int, List[str]]:
        """
        Get top words for each LDA topic
        
        Args:
            n_words: Number of top words per topic
            
        Returns:
            Dictionary mapping topic_id -> list of top words
        """
        if self.lda_model is None or self.bow_vectorizer is None:
            logger.warning("LDA model or BoW vectorizer not fitted")
            return {}
        
        feature_names = self.bow_vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics[topic_idx] = top_words
        
        return topics
    
    def print_topics(self, n_words: int = 10):
        """Print top words for each topic"""
        topics = self.get_top_words_per_topic(n_words)
        
        print("\n" + "="*80)
        print("DISCOVERED TOPICS (LDA)")
        print("="*80)
        
        for topic_id, words in topics.items():
            print(f"\nTopic {topic_id}: {', '.join(words)}")
        
        print("="*80 + "\n")


def aggregate_text_features_by_date(articles_df: pd.DataFrame,
                                     text_col: str = 'text',
                                     date_col: str = 'date',
                                     feature_extractor: Optional[RichTextFeatureExtractor] = None
                                     ) -> pd.DataFrame:
    """
    Aggregate text features by date for time series modeling
    
    Args:
        articles_df: DataFrame with articles (must have text and date columns)
        text_col: Column name containing text
        date_col: Column name containing dates
        feature_extractor: Fitted RichTextFeatureExtractor (if None, creates new one)
        
    Returns:
        DataFrame with daily aggregated text features
    """
    logger.info("Aggregating text features by date...")
    
    if feature_extractor is None:
        feature_extractor = RichTextFeatureExtractor()
        
        # Fit models on all texts
        texts = articles_df[text_col].fillna("").tolist()
        feature_extractor.fit_bow(texts)
        feature_extractor.fit_tfidf(texts)
        feature_extractor.fit_lda(texts)
    
    # Extract features for all articles
    texts = articles_df[text_col].fillna("").tolist()
    features_df = feature_extractor.extract_all_features(texts)
    
    # Add date column
    features_df['date'] = pd.to_datetime(articles_df[date_col]).dt.date
    
    # Aggregate by date (mean of all features)
    daily_features = features_df.groupby('date').mean().reset_index()
    
    logger.info(f"Aggregated to {len(daily_features)} days with {daily_features.shape[1]-1} features")
    
    return daily_features

