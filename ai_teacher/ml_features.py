"""
Machine Learning features for the AI Teacher application.
This module contains ML models and algorithms for personalized learning.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

from .models import Flashcard, StudySession
from .nlp_processor import NLPProcessor


class ContentSimilarityAnalyzer:
    """
    Analyzes content similarity between flashcards using ML techniques.
    This is a great starting point for learning ML concepts!
    """
    
    def __init__(self):
        """Initialize the content similarity analyzer."""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.nlp = NLPProcessor()
        self.similarity_matrix = None
        self.flashcard_texts = []
        self.flashcard_ids = []
    
    def prepare_data(self, flashcards: List[Flashcard]) -> np.ndarray:
        """
        Prepare flashcard data for similarity analysis.
        
        Args:
            flashcards: List of flashcards to analyze
            
        Returns:
            TF-IDF matrix of flashcard content
        """
        # Combine term and definition for each flashcard
        self.flashcard_texts = []
        self.flashcard_ids = []
        
        for card in flashcards:
            # Create a combined text representation
            combined_text = f"{card.term} {card.definition}"
            self.flashcard_texts.append(combined_text)
            self.flashcard_ids.append(card.term)  # Using term as ID
        
        # Vectorize the text
        tfidf_matrix = self.vectorizer.fit_transform(self.flashcard_texts)
        return tfidf_matrix.toarray()
    
    def calculate_similarity_matrix(self, flashcards: List[Flashcard]) -> np.ndarray:
        """
        Calculate similarity matrix between all flashcards.
        
        Args:
            flashcards: List of flashcards to analyze
            
        Returns:
            Similarity matrix (n x n) where each cell represents similarity
        """
        tfidf_matrix = self.prepare_data(flashcards)
        
        # Calculate cosine similarity
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return self.similarity_matrix
    
    def find_similar_flashcards(self, target_flashcard: Flashcard, 
                              flashcards: List[Flashcard], 
                              top_k: int = 5) -> List[Tuple[Flashcard, float]]:
        """
        Find flashcards similar to a target flashcard.
        
        Args:
            target_flashcard: The flashcard to find similarities for
            flashcards: List of all flashcards
            top_k: Number of similar flashcards to return
            
        Returns:
            List of tuples (similar_flashcard, similarity_score)
        """
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix(flashcards)
        
        # Find the index of the target flashcard
        try:
            target_idx = self.flashcard_ids.index(target_flashcard.term)
        except ValueError:
            return []
        
        # Get similarity scores for the target flashcard
        similarities = self.similarity_matrix[target_idx]
        
        # Get top-k similar flashcards (excluding the target itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for idx in similar_indices:
            if similarities[idx] > 0.1:  # Only include if similarity > 0.1
                similar_card = flashcards[idx]
                results.append((similar_card, similarities[idx]))
        
        return results
    
    def group_similar_flashcards(self, flashcards: List[Flashcard], 
                               n_clusters: int = 5) -> Dict[int, List[Flashcard]]:
        """
        Group flashcards into clusters based on content similarity.
        
        Args:
            flashcards: List of flashcards to cluster
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary mapping cluster_id to list of flashcards
        """
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix(flashcards)
        
        # Use K-means clustering on the TF-IDF matrix
        tfidf_matrix = self.vectorizer.transform(self.flashcard_texts)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Group flashcards by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(flashcards[i])
        
        return clusters
    
    def get_similarity_insights(self, flashcards: List[Flashcard]) -> Dict:
        """
        Get insights about content similarity in the flashcard collection.
        
        Args:
            flashcards: List of flashcards to analyze
            
        Returns:
            Dictionary with similarity insights
        """
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix(flashcards)
        
        # Calculate statistics
        similarities = self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]
        
        insights = {
            'total_flashcards': len(flashcards),
            'average_similarity': float(np.mean(similarities)),
            'max_similarity': float(np.max(similarities)),
            'min_similarity': float(np.min(similarities)),
            'highly_similar_pairs': int(np.sum(similarities > 0.7)),
            'unique_content_ratio': float(np.sum(similarities < 0.3) / len(similarities))
        }
        
        return insights


class DifficultyPredictor:
    """
    Predicts flashcard difficulty based on text features.
    This teaches you about feature engineering and regression!
    """
    
    def __init__(self):
        """Initialize the difficulty predictor."""
        self.model = LinearRegression()
        self.feature_names = []
        self.is_trained = False
        self.nlp = NLPProcessor()
    
    def extract_features(self, flashcard: Flashcard) -> List[float]:
        """
        Extract features from a flashcard for difficulty prediction.
        
        Args:
            flashcard: The flashcard to extract features from
            
        Returns:
            List of feature values
        """
        features = []
        
        # Text-based features
        combined_text = f"{flashcard.term} {flashcard.definition}"
        
        # Basic text features
        features.append(len(flashcard.term))  # Term length
        features.append(len(flashcard.definition))  # Definition length
        features.append(len(combined_text.split()))  # Total word count
        features.append(len(set(combined_text.lower().split())))  # Unique word count
        
        # Complexity features
        complexity = self.nlp.analyze_text_complexity(combined_text)
        features.append(complexity.get('avg_sentence_length', 0))
        features.append(complexity.get('avg_word_length', 0))
        features.append(complexity.get('lexical_diversity', 0))
        
        # Category encoding (simple one-hot for now)
        categories = ['general', 'programming', 'computer science', 'ai', 'machine learning']
        category_features = [1 if flashcard.category.lower() == cat else 0 for cat in categories]
        features.extend(category_features)
        
        return features
    
    def prepare_training_data(self, flashcards: List[Flashcard]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for the difficulty prediction model.
        
        Args:
            flashcards: List of flashcards with known difficulties
            
        Returns:
            Tuple of (features_matrix, difficulty_labels)
        """
        features_list = []
        difficulties = []
        
        for card in flashcards:
            features = self.extract_features(card)
            features_list.append(features)
            difficulties.append(card.difficulty)
        
        # Store feature names for later use
        if not self.feature_names:
            self.feature_names = [
                'term_length', 'definition_length', 'word_count', 'unique_words',
                'avg_sentence_length', 'avg_word_length', 'lexical_diversity'
            ] + [f'category_{cat}' for cat in ['general', 'programming', 'computer science', 'ai', 'machine learning']]
        
        return np.array(features_list), np.array(difficulties)
    
    def train(self, flashcards: List[Flashcard]) -> Dict:
        """
        Train the difficulty prediction model.
        
        Args:
            flashcards: List of flashcards with known difficulties
            
        Returns:
            Training metrics
        """
        X, y = self.prepare_training_data(flashcards)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'mse': mse,
            'r2_score': r2,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict_difficulty(self, flashcard: Flashcard) -> float:
        """
        Predict the difficulty of a flashcard.
        
        Args:
            flashcard: The flashcard to predict difficulty for
            
        Returns:
            Predicted difficulty score
        """
        if not self.is_trained:
            return 2.5  # Default difficulty
        
        features = self.extract_features(flashcard)
        features_array = np.array(features).reshape(1, -1)
        
        prediction = self.model.predict(features_array)[0]
        return max(1.0, min(4.0, prediction))  # Clamp between 1.0 and 4.0
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            return {}
        
        # For linear regression, we can use the coefficients as importance
        importance = {}
        for i, feature_name in enumerate(self.feature_names):
            importance[feature_name] = abs(self.model.coef_[i])
        
        return importance


# TODO: You can implement these next!
class StudyRecommendationEngine:
    """
    Recommends optimal study sessions based on user behavior patterns.
    This is where you'll learn about recommendation systems!
    """
    
    def __init__(self):
        """Initialize the recommendation engine."""
        pass
    
    # TODO: Implement this method
    def recommend_study_session(self, user_history: List[StudySession], 
                              available_cards: List[Flashcard]) -> List[Flashcard]:
        """
        Recommend flashcards for the next study session.
        
        Args:
            user_history: List of previous study sessions
            available_cards: List of flashcards available for study
            
        Returns:
            Recommended flashcards for study
        """
        # Your implementation here!
        # Hint: Consider factors like:
        # - Cards that are due for review
        # - User's performance patterns
        # - Time since last study
        # - Difficulty progression
        pass


class PerformanceAnalyzer:
    """
    Analyzes learning performance and predicts learning curves.
    This teaches you about time series analysis and performance metrics!
    """
    
    def __init__(self):
        """Initialize the performance analyzer."""
        pass
    
    # TODO: Implement this method
    def analyze_learning_curve(self, study_sessions: List[StudySession]) -> Dict:
        """
        Analyze the user's learning curve over time.
        
        Args:
            study_sessions: List of study sessions to analyze
            
        Returns:
            Dictionary with learning curve analysis
        """
        # Your implementation here!
        # Hint: Consider metrics like:
        # - Accuracy trends over time
        # - Study session frequency
        # - Performance by category
        # - Learning velocity
        pass
