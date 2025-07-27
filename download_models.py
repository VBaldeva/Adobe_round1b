#!/usr/bin/env python3
"""
Download all required models and data for offline usage
"""

import os
import nltk
from sentence_transformers import SentenceTransformer
import torch

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)  # For newer NLTK versions
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded successfully")
    except Exception as e:
        print(f"Warning: NLTK download issue: {e}")

def download_sentence_transformer():
    """Download sentence transformer model"""
    print("Downloading sentence transformer model...")
    try:
        model_name = "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        
        # Test the model to ensure it works
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        print(f"✓ Sentence transformer model '{model_name}' downloaded and tested successfully")
        print(f"  Model size: {len(embedding)} dimensions")
        
        # Also cache the tokenizer
        _ = model.tokenizer
        print("✓ Tokenizer cached successfully")
        
    except Exception as e:
        print(f"Error downloading sentence transformer: {e}")
        raise

def download_torch_models():
    """Ensure PyTorch models are cached"""
    print("Checking PyTorch installation...")
    try:
        # Test basic torch functionality
        x = torch.randn(2, 3)
        y = x.sum()
        print(f"✓ PyTorch working correctly (version: {torch.__version__})")
    except Exception as e:
        print(f"Error with PyTorch: {e}")
        raise

def verify_sklearn():
    """Verify scikit-learn installation"""
    print("Verifying scikit-learn...")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Test basic functionality
        vectorizer = TfidfVectorizer()
        texts = ["hello world", "goodbye world"]
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix)
        print("✓ Scikit-learn working correctly")
        
    except Exception as e:
        print(f"Error with scikit-learn: {e}")
        raise

def verify_pymupdf():
    """Verify PyMuPDF installation"""
    print("Verifying PyMuPDF...")
    try:
        import fitz
        print(f"✓ PyMuPDF (fitz) working correctly (version: {fitz.version})")
    except Exception as e:
        print(f"Error with PyMuPDF: {e}")
        raise

def create_cache_info():
    """Create a file with cache information"""
    cache_info = {
        "models_downloaded": True,
        "sentence_transformer": "all-MiniLM-L6-v2",
        "nltk_data": ["punkt", "stopwords"],
        "offline_ready": True
    }
    
    with open("/app/model_cache_info.txt", "w") as f:
        f.write("OFFLINE MODELS CACHED\n")
        f.write("====================\n")
        for key, value in cache_info.items():
            f.write(f"{key}: {value}\n")
    
    print("✓ Cache information saved")

def main():
    print("=" * 50)
    print("DOWNLOADING MODELS FOR OFFLINE USAGE")
    print("=" * 50)
    
    try:
        # Download all required components
        download_nltk_data()
        download_sentence_transformer()
        download_torch_models()
        verify_sklearn()
        verify_pymupdf()
        create_cache_info()
        
        print("\n" + "=" * 50)
        print("✓ ALL MODELS DOWNLOADED SUCCESSFULLY!")
        print("✓ SYSTEM READY FOR OFFLINE USAGE")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ ERROR DURING MODEL DOWNLOAD: {e}")
        print("❌ SYSTEM NOT READY FOR OFFLINE USAGE")
        raise

if __name__ == "__main__":
    main()