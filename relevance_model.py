from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

class RelevanceModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"[INFO] Loading sentence transformer model: {model_name}")
        try:
            # For offline usage, disable online model fetching
            self.model = SentenceTransformer(model_name, cache_folder='/root/.cache/torch/sentence_transformers')
            print(f"[INFO] Model loaded successfully")
        except Exception as e:
            print(f"[WARNING] Failed to load model {model_name}: {e}")
            print("[INFO] Falling back to TF-IDF only mode")
            self.model = None
            
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),  # Include bigrams
            min_df=1,
            max_df=0.95
        )

    def extract_keywords(self, text):
        """Extract important keywords and phrases"""
        # Remove common academic phrases
        text = re.sub(r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b', ' ', text.lower())
        
        # Extract potential keywords
        keywords = []
        
        # Look for domain-specific terms
        domain_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Title case terms
            r'\b\w+(?:ology|ics|ism|tion|sion|ment|ness|ity)\b',  # Academic suffixes
            r'\b(?:method|approach|technique|algorithm|model|framework)\w*\b',  # Technical terms
        ]
        
        for pattern in domain_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend(matches)
        
        return ' '.join(keywords[:10])  # Limit to top keywords

    def calculate_position_bonus(self, sections):
        """Give bonus to sections appearing earlier in documents"""
        bonuses = {}
        for i, section in enumerate(sections):
            # Earlier sections get higher bonus
            page_bonus = max(0, (10 - section.get('page', 10)) / 10 * 0.1)
            # First few sections get additional bonus
            order_bonus = max(0, (len(sections) - i) / len(sections) * 0.05)
            bonuses[i] = page_bonus + order_bonus
        return bonuses

    def score_sections(self, query, sections):
        if not sections:
            return []
            
        # Prepare texts for scoring
        corpus = []
        clean_sections = []
        enhanced_texts = []

        for section in sections:
            heading = section.get("section_title", section.get("text", "")).strip()
            summary = section.get("summary", section.get("context", "")).strip()
            
            if not heading:
                continue

            # Create enhanced text for better matching
            keywords = self.extract_keywords(f"{heading} {summary}")
            enhanced_text = f"{heading}. {summary} {keywords}".strip()
            
            corpus.append(enhanced_text)
            enhanced_texts.append(enhanced_text)
            clean_sections.append(section)

        if not corpus:
            return []

        # Calculate position bonuses
        position_bonuses = self.calculate_position_bonus(clean_sections)

        try:
            # TF-IDF scoring
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            query_vec = self.vectorizer.transform([query])
            tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        except:
            tfidf_scores = np.zeros(len(corpus))

        try:
            # Semantic embedding scoring (only if model is available)
            if self.model:
                query_emb = self.model.encode(query, convert_to_tensor=True)
                corpus_emb = self.model.encode(corpus, convert_to_tensor=True)
                semantic_scores = util.cos_sim(query_emb, corpus_emb)[0].cpu().numpy()
            else:
                print("[INFO] Using TF-IDF only (no semantic model)")
                semantic_scores = tfidf_scores  # Fallback to TF-IDF
        except Exception as e:
            print(f"[WARNING] Semantic scoring failed: {e}")
            semantic_scores = tfidf_scores  # Fallback to TF-IDF

        # Keyword matching bonus
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        results = []
        for idx, section in enumerate(clean_sections):
            # Base scores
            tfidf_score = float(tfidf_scores[idx]) if idx < len(tfidf_scores) else 0.0
            semantic_score = float(semantic_scores[idx]) if idx < len(semantic_scores) else 0.0
            
            # Keyword matching bonus
            section_text = enhanced_texts[idx].lower()
            section_words = set(re.findall(r'\w+', section_text))
            keyword_overlap = len(query_words.intersection(section_words)) / max(len(query_words), 1)
            keyword_bonus = keyword_overlap * 0.2
            
            # Position bonus
            position_bonus = position_bonuses.get(idx, 0)
            
            # Heading quality bonus (prefer clear, descriptive headings)
            heading = section.get("section_title", section.get("text", ""))
            heading_bonus = 0
            if len(heading.split()) >= 2:  # Multi-word headings
                heading_bonus += 0.05
            if any(word in heading.lower() for word in ['method', 'result', 'analysis', 'approach', 'technique']):
                heading_bonus += 0.1
                
            # Content length bonus (prefer sections with substantial content)
            content_length = len(section.get("summary", section.get("context", "")).split())
            length_bonus = min(content_length / 100, 0.1)  # Cap at 0.1
            
            # Combine scores with weights
            total_score = (
                0.4 * semantic_score +      # Semantic similarity (highest weight)
                0.3 * tfidf_score +         # TF-IDF relevance
                0.1 * keyword_bonus +       # Direct keyword matching
                0.1 * position_bonus +      # Position in document
                0.05 * heading_bonus +      # Heading quality
                0.05 * length_bonus         # Content length
            )
            
            results.append((total_score, section))

        # Sort by score (highest first)
        results.sort(reverse=True, key=lambda x: x[0])
        
        return results

    def filter_duplicates(self, ranked_results, similarity_threshold=0.8):
        """Remove very similar sections"""
        if not ranked_results:
            return ranked_results
            
        filtered = []
        seen_texts = []
        
        for score, section in ranked_results:
            heading = section.get("section_title", section.get("text", "")).lower().strip()
            
            # Check for similarity with already selected sections
            is_duplicate = False
            for seen_text in seen_texts:
                # Simple similarity check
                words1 = set(heading.split())
                words2 = set(seen_text.split())
                if words1 and words2:
                    overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                    if overlap > similarity_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append((score, section))
                seen_texts.append(heading)
                
        return filtered