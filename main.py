from models.baseline_nbow import baseline_nbow
from models.sentiment_model import Sentiment_Analysis

Sentiment_Analysis("datastories.twitter", 300, 50, PERSIST=False)