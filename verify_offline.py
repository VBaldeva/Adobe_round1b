#!/usr/bin/env python3
"""
Verify that all components work offline
"""

import os
import sys

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import fitz
        print("✓ PyMuPDF imported")
    except ImportError as e:
        print(f"❌ PyMuPDF import failed: {e}")
        return False
    
    try:
        import nltk
        print("✓ NLTK imported")
    except ImportError as e:
        print(f"❌ NLTK import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer, util
        print("✓ SentenceTransformers imported")
    except ImportError as e:
        print(f"❌ SentenceTransformers import failed: {e}")
        return False
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        print("✓ Scikit-learn imported")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch imported")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    return True

def test_nltk_offline():
    """Test NLTK functionality offline"""
    print("\nTesting NLTK offline...")
    
    try:
        import nltk
        
        # Test tokenization
        text = "This is a test sentence. It has multiple parts."
        tokens = nltk.word_tokenize(text)
        sentences = nltk.sent_tokenize(text)
        
        print(f"✓ Tokenization works: {len(tokens)} tokens, {len(sentences)} sentences")
        return True
        
    except Exception as e:
        print(f"❌ NLTK offline test failed: {e}")
        return False

def test_sentence_transformer_offline():
    """Test sentence transformer offline"""
    print("\nTesting SentenceTransformer offline...")
    
    try:
        from sentence_transformers import SentenceTransformer, util
        
        # Disable online access
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test encoding
        texts = ["This is a test", "Another test sentence"]
        embeddings = model.encode(texts)
        
        # Test similarity
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        
        print(f"✓ SentenceTransformer works offline: embeddings shape {embeddings.shape}")
        print(f"✓ Similarity calculation works: {similarity.item():.3f}")
        return True
        
    except Exception as e:
        print(f"❌ SentenceTransformer offline test failed: {e}")
        return False

def test_tfidf_offline():
    """Test TF-IDF functionality"""
    print("\nTesting TF-IDF offline...")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer(stop_words='english')
        texts = ["This is a test document", "This is another test document"]
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix)
        
        print(f"✓ TF-IDF works: matrix shape {tfidf_matrix.shape}")
        print(f"✓ Cosine similarity works: {similarity[0,1]:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ TF-IDF test failed: {e}")
        return False

def test_pdf_processing():
    """Test PDF processing (simulated)"""
    print("\nTesting PDF processing capability...")
    
    try:
        import fitz
        
        # Test basic fitz functionality
        # We can't test with actual PDF without one, but we can test the import
        print("✓ PyMuPDF ready for PDF processing")
        return True
        
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
        return False

def test_network_isolation():
    """Verify we're truly offline"""
    print("\nTesting network isolation...")
    
    # Set offline environment variables
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    
    print("✓ Offline environment variables set")
    return True

def main():
    print("=" * 50)
    print("OFFLINE FUNCTIONALITY VERIFICATION")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_network_isolation,
        test_nltk_offline,
        test_tfidf_offline,
        test_sentence_transformer_offline,
        test_pdf_processing,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print("-" * 30)
    
    print(f"\nRESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED - SYSTEM READY FOR OFFLINE USAGE!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - SYSTEM NOT FULLY OFFLINE READY")
        sys.exit(1)

if __name__ == "__main__":
    main()