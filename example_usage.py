"""
Example usage of the AI Teacher application.
This demonstrates how to use the AI Teacher programmatically.
"""

import sys
import os

# Add the ai_teacher module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_teacher'))

from ai_teacher.main import AITeacher


def example_usage():
    """Demonstrate how to use the AI Teacher programmatically."""
    print("📚 AI Teacher Example Usage")
    print("=" * 40)
    
    # Initialize AI Teacher
    ai_teacher = AITeacher("example_ai_teacher.db")
    
    # Example 1: Create flashcards manually
    print("\n1. Creating flashcards manually...")
    ai_teacher.create_flashcard("Python", "A high-level programming language", "Programming")
    ai_teacher.create_flashcard("Algorithm", "A step-by-step procedure for solving a problem", "Computer Science")
    ai_teacher.create_flashcard("Database", "A structured collection of data", "Computer Science")
    
    # Example 2: Add study notes
    print("\n2. Adding study notes...")
    ai_teacher.add_note(
        "Python Basics",
        "Python is an interpreted, high-level programming language. It has a simple syntax and is great for beginners. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        "Programming"
    )
    
    # Example 3: Generate flashcards from text
    print("\n3. Generating flashcards from text...")
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. 
    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. 
    Supervised learning uses labeled training data to make predictions. Unsupervised learning finds patterns in data without labels. 
    Reinforcement learning learns through interaction with an environment and receiving rewards or penalties.
    """
    
    ai_teacher.create_flashcards_from_text(sample_text, "Machine Learning")
    
    # Example 4: Show progress
    print("\n4. Current progress...")
    ai_teacher.show_progress()
    
    # Example 5: View all flashcards
    print("\n5. All flashcards...")
    ai_teacher._view_flashcards()
    
    print("\n✅ Example completed! You can now run study sessions or add more content.")
    print("\nTo start an interactive session, run:")
    print("python -m ai_teacher.main")


if __name__ == "__main__":
    example_usage()
