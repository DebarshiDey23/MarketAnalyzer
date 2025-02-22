import torch
from transformers import BertTokenizer, BertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# Initialize VADER for social media sentiment
vader = SentimentIntensityAnalyzer()

# Load FinBERT model for financial sentiment
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(finbert_model_name)
finbert_model = BertForSequenceClassification.from_pretrained(finbert_model_name)
finbert_model.eval()  # Set to evaluation mode

def get_vader_sentiment(text):
    """
    Uses VADER to analyze sentiment of social media text.
    Returns a sentiment score from -1 (negative) to +1 (positive).
    """
    sentiment = vader.polarity_scores(text)
    return sentiment["compound"]  # Compound score represents overall sentiment

def get_finbert_sentiment(text):
    """
    Uses FinBERT to analyze financial news sentiment.
    Returns a score between -1 (negative) to +1 (positive).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs).logits
    scores = torch.nn.functional.softmax(outputs, dim=1).numpy()[0]

    # FinBERT gives scores for ['negative', 'neutral', 'positive']
    sentiment_score = scores[2] - scores[0]  # Positive - Negative
    return sentiment_score

def analyze_sentiment(text, source="social"):
    """
    Determines which sentiment model to use based on the source.
    - 'social' (e.g., Twitter) → Uses VADER
    - 'news' (e.g., Yahoo Finance) → Uses FinBERT
    """
    if source == "social":
        return get_vader_sentiment(text)
    elif source == "news":
        return get_finbert_sentiment(text)
    else:
        raise ValueError("Source must be 'social' or 'news'")