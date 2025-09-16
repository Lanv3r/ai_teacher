"""
Natural Language Processing module for the AI Teacher application.
Handles text processing, summarization, and flashcard generation.
"""

import re
import spacy
from typing import List, Dict, Tuple, Optional
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class NLPProcessor:
    """
    Handles natural language processing tasks for the AI Teacher application.
    """
    
    def __init__(self):
        """Initialize the NLP processor with required models."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Some features may not work.")
            self.nlp = None
        
        # Initialize NLTK components
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except LookupError:
            print("Warning: NLTK data not found. Some features may not work.")
            self.stop_words = set()
            self.lemmatizer = None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for analysis.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        
        return text
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract important keywords from text using TF-IDF.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords sorted by importance
        """
        if not text.strip():
            return []
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Split into sentences - use simple fallback if NLTK not available
        try:
            sentences = sent_tokenize(cleaned_text)
        except LookupError:
            # Fallback: simple sentence splitting
            sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
            sentences = [s + '.' for s in sentences if not s.endswith('.')]
        
        if len(sentences) < 2:
            # For single sentence, use simple word frequency
            try:
                words = word_tokenize(cleaned_text.lower())
            except LookupError:
                # Fallback: simple word splitting
                words = cleaned_text.lower().split()
            words = [word for word in words if word.isalpha() and word not in self.stop_words]
            word_freq = Counter(words)
            return [word for word, _ in word_freq.most_common(max_keywords)]
        
        # Use TF-IDF for multiple sentences
        try:
            vectorizer = TfidfVectorizer(
                max_features=max_keywords * 2,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top keywords
            top_indices = np.argsort(mean_scores)[-max_keywords:][::-1]
            keywords = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
            
            return keywords
            
        except Exception as e:
            print(f"Error in keyword extraction: {e}")
            # Fallback to simple word frequency
            words = word_tokenize(cleaned_text.lower())
            words = [word for word in words if word.isalpha() and word not in self.stop_words]
            word_freq = Counter(words)
            return [word for word, _ in word_freq.most_common(max_keywords)]
    
    def summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """
        Summarize text by extracting the most important sentences.
        
        Args:
            text: Text to summarize
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Summarized text
        """
        if not text.strip():
            return ""
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Split into sentences - use simple fallback if NLTK not available
        try:
            sentences = sent_tokenize(cleaned_text)
        except LookupError:
            # Fallback: simple sentence splitting
            sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
            sentences = [s + '.' for s in sentences if not s.endswith('.')]
        
        if len(sentences) <= max_sentences:
            return cleaned_text
        
        try:
            # Use TF-IDF to score sentences
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores
            sentence_scores = np.mean(tfidf_matrix.toarray(), axis=1)
            
            # Get top sentences
            top_indices = np.argsort(sentence_scores)[-max_sentences:][::-1]
            top_indices = sorted(top_indices)  # Maintain original order
            
            summary_sentences = [sentences[i] for i in top_indices]
            return ' '.join(summary_sentences)
            
        except Exception as e:
            print(f"Error in text summarization: {e}")
            # Fallback: return first few sentences
            return ' '.join(sentences[:max_sentences])
    
    def generate_flashcards_from_text(self, text: str, max_cards: int = 5) -> List[Dict[str, str]]:
        """
        Generate flashcards from text content.
        
        Args:
            text: Text to generate flashcards from
            max_cards: Maximum number of flashcards to generate
            
        Returns:
            List of flashcard dictionaries with 'term' and 'definition' keys
        """
        if not text.strip():
            return []
        
        flashcards = []
        
        # Extract keywords
        keywords = self.extract_keywords(text, max_cards * 2)
        
        # Clean text for processing
        cleaned_text = self.clean_text(text)
        try:
            sentences = sent_tokenize(cleaned_text)
        except LookupError:
            # Fallback: simple sentence splitting
            sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
            sentences = [s + '.' for s in sentences if not s.endswith('.')]
        
        # Generate flashcards for each keyword
        for keyword in keywords[:max_cards]:
            # Find sentences containing the keyword
            relevant_sentences = [sent for sent in sentences 
                                if keyword.lower() in sent.lower()]
            
            if relevant_sentences:
                # Use the first relevant sentence as definition
                definition = relevant_sentences[0]
                
                # Clean up the definition
                definition = definition.strip()
                if len(definition) > 200:
                    definition = definition[:200] + "..."
                
                flashcards.append({
                    'term': keyword.title(),
                    'definition': definition
                })
        
        return flashcards
    
    def extract_concepts(self, text: str) -> List[Dict[str, str]]:
        """
        Extract key concepts and their definitions from text.
        
        Args:
            text: Text to extract concepts from
            
        Returns:
            List of concept dictionaries with 'concept' and 'definition' keys
        """
        if not self.nlp:
            return []
        
        concepts = []
        doc = self.nlp(text)
        
        # Extract noun phrases as potential concepts
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # Multi-word phrases
                concept = chunk.text
                
                # Find sentences containing this concept
                relevant_sentences = [sent.text for sent in doc.sents 
                                    if concept.lower() in sent.text.lower()]
                
                if relevant_sentences:
                    # Use the first sentence as definition
                    definition = relevant_sentences[0].strip()
                    if len(definition) > 150:
                        definition = definition[:150] + "..."
                    
                    concepts.append({
                        'concept': concept,
                        'definition': definition
                    })
        
        return concepts[:10]  # Limit to 10 concepts
    
    def analyze_text_complexity(self, text: str) -> Dict[str, float]:
        """
        Analyze the complexity of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with complexity metrics
        """
        if not text.strip():
            return {}
        
        # Basic metrics
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback: simple sentence splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            sentences = [s + '.' for s in sentences if not s.endswith('.')]
        
        try:
            words = word_tokenize(text.lower())
        except LookupError:
            # Fallback: simple word splitting
            words = text.lower().split()
        words = [word for word in words if word.isalpha()]
        
        # Calculate metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Calculate lexical diversity (unique words / total words)
        unique_words = len(set(words))
        lexical_diversity = unique_words / len(words) if words else 0
        
        # Calculate readability score (simplified Flesch Reading Ease)
        # Higher score = easier to read
        if sentences and words:
            readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        else:
            readability = 0
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'lexical_diversity': lexical_diversity,
            'readability_score': readability,
            'complexity_level': self._get_complexity_level(readability)
        }
    
    def _get_complexity_level(self, readability_score: float) -> str:
        """Convert readability score to complexity level."""
        if readability_score >= 80:
            return "Very Easy"
        elif readability_score >= 60:
            return "Easy"
        elif readability_score >= 40:
            return "Medium"
        elif readability_score >= 20:
            return "Hard"
        else:
            return "Very Hard"
    
    def find_similar_content(self, text1: str, text2: str) -> float:
        """
        Find similarity between two texts using cosine similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1.strip() or not text2.strip():
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
