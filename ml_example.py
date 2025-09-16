"""
Example of using ML features in AI Teacher.
This demonstrates the content similarity analysis - perfect for learning ML basics!
"""

import sys
import os

# Add the ai_teacher module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_teacher'))

from ai_teacher.main import AITeacher
from ai_teacher.ml_features import ContentSimilarityAnalyzer, DifficultyPredictor


def demonstrate_content_similarity():
    """Demonstrate content similarity analysis - your first ML feature!"""
    print("🔍 Content Similarity Analysis Demo")
    print("=" * 50)
    
    # Initialize AI Teacher and add some sample data
    ai_teacher = AITeacher("ml_demo.db")
    
    # Add some sample flashcards
    sample_flashcards = [
        ("Machine Learning", "A subset of AI that enables computers to learn from data", "AI"),
        ("Deep Learning", "A subset of machine learning using neural networks", "AI"),
        ("Neural Network", "A computing system inspired by biological neural networks", "AI"),
        ("Python", "A high-level programming language", "Programming"),
        ("Algorithm", "A step-by-step procedure for solving problems", "Computer Science"),
        ("Data Structure", "A way of organizing data in computer memory", "Computer Science"),
        ("Supervised Learning", "Machine learning with labeled training data", "AI"),
        ("Unsupervised Learning", "Machine learning that finds patterns in unlabeled data", "AI"),
    ]
    
    print("📚 Adding sample flashcards...")
    for term, definition, category in sample_flashcards:
        ai_teacher.create_flashcard(term, definition, category)
    
    # Get all flashcards
    flashcards = ai_teacher.db.get_all_flashcards()
    
    # Initialize the similarity analyzer
    similarity_analyzer = ContentSimilarityAnalyzer()
    
    print("\n🔍 Analyzing content similarity...")
    
    # Calculate similarity matrix
    similarity_matrix = similarity_analyzer.calculate_similarity_matrix(flashcards)
    print(f"✅ Calculated similarity matrix: {similarity_matrix.shape}")
    
    # Find similar flashcards for "Machine Learning"
    target_card = next(card for card in flashcards if card.term == "Machine Learning")
    similar_cards = similarity_analyzer.find_similar_flashcards(target_card, flashcards, top_k=3)
    
    print(f"\n🎯 Flashcards similar to '{target_card.term}':")
    for card, similarity in similar_cards:
        print(f"  • {card.term}: {similarity:.3f} similarity")
    
    # Group flashcards into clusters
    print(f"\n📊 Grouping flashcards into clusters...")
    clusters = similarity_analyzer.group_similar_flashcards(flashcards, n_clusters=3)
    
    for cluster_id, cluster_cards in clusters.items():
        print(f"\nCluster {cluster_id}:")
        for card in cluster_cards:
            print(f"  • {card.term} ({card.category})")
    
    # Get similarity insights
    insights = similarity_analyzer.get_similarity_insights(flashcards)
    print(f"\n📈 Similarity Insights:")
    print(f"  • Average similarity: {insights['average_similarity']:.3f}")
    print(f"  • Highly similar pairs: {insights['highly_similar_pairs']}")
    print(f"  • Unique content ratio: {insights['unique_content_ratio']:.3f}")


def demonstrate_difficulty_prediction():
    """Demonstrate difficulty prediction - your second ML feature!"""
    print("\n\n🎯 Difficulty Prediction Demo")
    print("=" * 50)
    
    # Initialize the difficulty predictor
    predictor = DifficultyPredictor()
    
    # Get flashcards from the database
    ai_teacher = AITeacher("ml_demo.db")
    flashcards = ai_teacher.db.get_all_flashcards()
    
    if len(flashcards) < 5:
        print("❌ Need at least 5 flashcards to train the model")
        return
    
    print("🤖 Training difficulty prediction model...")
    
    # Train the model
    training_metrics = predictor.train(flashcards)
    print(f"✅ Model trained!")
    print(f"  • R² Score: {training_metrics['r2_score']:.3f}")
    print(f"  • Mean Squared Error: {training_metrics['mse']:.3f}")
    
    # Show feature importance
    feature_importance = predictor.get_feature_importance()
    print(f"\n📊 Feature Importance (top 5):")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:5]:
        print(f"  • {feature}: {importance:.3f}")
    
    # Predict difficulty for a new flashcard
    print(f"\n🔮 Predicting difficulty for new flashcards...")
    test_cards = [
        ("Quantum Computing", "Computing using quantum mechanical phenomena", "Computer Science"),
        ("Hello World", "A simple program that outputs 'Hello, World!'", "Programming"),
        ("Convolutional Neural Network", "A deep learning architecture for image processing", "AI")
    ]
    
    for term, definition, category in test_cards:
        test_card = ai_teacher.create_flashcard(term, definition, category)
        predicted_difficulty = predictor.predict_difficulty(test_card)
        print(f"  • {term}: {predicted_difficulty:.2f} difficulty")


def your_turn_to_implement():
    """This is where YOU get to implement ML features!"""
    print("\n\n🚀 Your Turn to Implement ML Features!")
    print("=" * 50)
    
    print("""
    Here are some ML features YOU can implement:
    
    1. 📊 Study Recommendation Engine
       - Implement recommend_study_session() in StudyRecommendationEngine
       - Consider factors like due cards, user performance, time patterns
    
    2. 📈 Performance Analysis
       - Implement analyze_learning_curve() in PerformanceAnalyzer
       - Track accuracy trends, study frequency, category performance
    
    3. 🎯 Advanced Similarity Features
       - Add semantic similarity using word embeddings
       - Implement topic modeling for flashcard categorization
    
    4. 🧠 Personalized Learning Paths
       - Create adaptive difficulty adjustment
       - Implement spaced repetition optimization
    
    Start with the StudyRecommendationEngine - it's the most practical!
    """)
    
    # TODO: Your implementation goes here!
    # Hint: Start by implementing recommend_study_session() method


if __name__ == "__main__":
    print("🤖 AI Teacher ML Features Demo")
    print("=" * 60)
    
    # Demo the implemented features
    demonstrate_content_similarity()
    demonstrate_difficulty_prediction()
    
    # Show what you can implement
    your_turn_to_implement()
    
    print("\n" + "=" * 60)
    print("🎓 ML Learning Path Complete!")
    print("\nNext steps:")
    print("1. Implement the StudyRecommendationEngine")
    print("2. Add the PerformanceAnalyzer")
    print("3. Experiment with different ML algorithms")
    print("4. Add more sophisticated features!")
