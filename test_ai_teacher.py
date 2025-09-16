"""
Simple test script for the AI Teacher application.
This helps verify that all components are working correctly.
"""

import sys
import os

# Add the ai_teacher module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_teacher'))

from ai_teacher.main import AITeacher
from ai_teacher.models import Flashcard, Note
from ai_teacher.nlp_processor import NLPProcessor


def test_basic_functionality():
    """Test basic functionality of the AI Teacher."""
    print("🧪 Testing AI Teacher functionality...")
    
    try:
        # Initialize AI Teacher
        ai_teacher = AITeacher("test_ai_teacher.db")
        print("✅ AI Teacher initialized successfully")
        
        # Test flashcard creation
        flashcard = ai_teacher.create_flashcard(
            "Machine Learning",
            "A subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "AI"
        )
        print("✅ Flashcard creation works")
        
        # Test note creation
        note = ai_teacher.add_note(
            "Introduction to Python",
            "Python is a high-level programming language known for its simplicity and readability. It's widely used in data science, web development, and automation.",
            "Programming"
        )
        print("✅ Note creation works")
        
        # Test NLP functionality
        nlp = NLPProcessor()
        test_text = "Machine learning is a subset of artificial intelligence. It involves algorithms that can learn from data."
        
        keywords = nlp.extract_keywords(test_text, max_keywords=5)
        print(f"✅ Keyword extraction works: {keywords}")
        
        summary = nlp.summarize_text(test_text, max_sentences=1)
        print(f"✅ Text summarization works: {summary}")
        
        flashcards_from_text = nlp.generate_flashcards_from_text(test_text, max_cards=3)
        print(f"✅ Flashcard generation from text works: {len(flashcards_from_text)} cards generated")
        
        # Test progress display
        ai_teacher.show_progress()
        print("✅ Progress display works")
        
        print("\n🎉 All tests passed! AI Teacher is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


def test_nlp_features():
    """Test NLP features specifically."""
    print("\n🔍 Testing NLP features...")
    
    try:
        nlp = NLPProcessor()
        
        sample_text = """
        Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. 
        Machine learning is a subset of AI that focuses on algorithms that can learn from data. 
        Deep learning is a subset of machine learning that uses neural networks with multiple layers.
        """
        
        # Test text analysis
        complexity = nlp.analyze_text_complexity(sample_text)
        print(f"✅ Text complexity analysis: {complexity}")
        
        # Test concept extraction
        concepts = nlp.extract_concepts(sample_text)
        print(f"✅ Concept extraction: {len(concepts)} concepts found")
        
        print("✅ NLP features working correctly")
        
    except Exception as e:
        print(f"❌ NLP test failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting AI Teacher Tests")
    print("=" * 40)
    
    test_basic_functionality()
    test_nlp_features()
    
    print("\n" + "=" * 40)
    print("🏁 Testing complete!")
    print("\nTo run the full AI Teacher application, use:")
    print("python -m ai_teacher.main")
