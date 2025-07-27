import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

nltk.download('punkt', quiet=True)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def summarize_text(text, max_sentences=3):
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return clean_text(text)

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)

    # Compute importance of each sentence (sum of TF-IDF weights)
    importance = X.sum(axis=1).A1
    ranked_indices = np.argsort(importance)[::-1]

    # Select top N sentences
    top_sentences = [sentences[i] for i in sorted(ranked_indices[:max_sentences])]
    return clean_text(" ".join(top_sentences))
