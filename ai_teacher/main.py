"""
Main AI Teacher application.
This is the entry point for the AI educational tool.
"""

import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional

from .models import Flashcard, Note, StudySession, LearningProgress
from .database import DatabaseManager
from .spaced_repetition import SpacedRepetition, StudyPlanner
from .nlp_processor import NLPProcessor


class AITeacher:
    """
    Main AI Teacher application class.
    Coordinates all components to provide an intelligent learning experience.
    """
    
    def __init__(self, db_path: str = "ai_teacher.db"):
        """Initialize the AI Teacher application."""
        self.db = DatabaseManager(db_path)
        self.sr = SpacedRepetition()
        self.planner = StudyPlanner(self.sr)
        self.nlp = NLPProcessor()
        
        print("🤖 AI Teacher initialized successfully!")
        print("📚 Ready to help you learn smarter!")
    
    def create_flashcard(self, term: str, definition: str, category: str = "general") -> Flashcard:
        """
        Create a new flashcard.
        
        Args:
            term: The term to learn
            definition: The definition of the term
            category: Category for organization
            
        Returns:
            Created flashcard
        """
        flashcard = Flashcard(
            term=term.strip(),
            definition=definition.strip(),
            category=category
        )
        
        # Save to database
        card_id = self.db.save_flashcard(flashcard)
        print(f"✅ Created flashcard: {term}")
        
        return flashcard
    
    def create_flashcards_from_text(self, text: str, category: str = "general") -> List[Flashcard]:
        """
        Automatically generate flashcards from text content.
        
        Args:
            text: Text to generate flashcards from
            category: Category for the flashcards
            
        Returns:
            List of created flashcards
        """
        print("🔍 Analyzing text and generating flashcards...")
        
        # Generate flashcards using NLP
        flashcard_data = self.nlp.generate_flashcards_from_text(text, max_cards=10)
        
        flashcards = []
        for data in flashcard_data:
            flashcard = Flashcard(
                term=data['term'],
                definition=data['definition'],
                category=category
            )
            self.db.save_flashcard(flashcard)
            flashcards.append(flashcard)
        
        print(f"✅ Generated {len(flashcards)} flashcards from text")
        return flashcards
    
    def add_note(self, title: str, content: str, category: str = "general") -> Note:
        """
        Add a new study note.
        
        Args:
            title: Note title
            content: Note content
            category: Category for organization
            
        Returns:
            Created note
        """
        # Generate summary using NLP
        summary = self.nlp.summarize_text(content, max_sentences=2)
        
        note = Note(
            title=title.strip(),
            content=content.strip(),
            category=category,
            summary=summary
        )
        
        # Save to database
        self.db.save_note(note)
        print(f"✅ Added note: {title}")
        
        return note
    
    def start_study_session(self, max_cards: int = 20) -> StudySession:
        """
        Start a new study session.
        
        Args:
            max_cards: Maximum number of cards to study
            
        Returns:
            Study session object
        """
        # Get flashcards for review
        all_cards = self.db.get_all_flashcards()
        session_cards = self.planner.create_study_session(all_cards, max_cards)
        
        if not session_cards:
            print("🎉 No cards to study right now! Great job!")
            return None
        
        # Create study session
        session = StudySession(
            session_id=str(uuid.uuid4()),
            categories=list(set(card.category for card in session_cards))
        )
        
        print(f"📖 Starting study session with {len(session_cards)} cards")
        print("=" * 50)
        
        # Study each card
        for i, card in enumerate(session_cards, 1):
            print(f"\n📝 Card {i}/{len(session_cards)}")
            print(f"Category: {card.category}")
            print(f"Term: {card.term}")
            
            # Show definition
            input("Press Enter to see the definition...")
            print(f"Definition: {card.definition}")
            
            # Get user feedback
            while True:
                try:
                    quality = int(input("\nHow well did you know this? (0-5): "))
                    if 0 <= quality <= 5:
                        break
                    else:
                        print("Please enter a number between 0 and 5")
                except ValueError:
                    print("Please enter a valid number")
            
            # Update card based on performance
            self.sr.calculate_next_review(card, quality)
            self.db.update_flashcard(card.term, card)  # Note: This needs card ID
            
            # Update session statistics
            session.cards_studied += 1
            if quality >= 3:
                session.correct_answers += 1
            else:
                session.incorrect_answers += 1
            
            print(f"✅ Card updated! Next review: {card.next_review.strftime('%Y-%m-%d') if card.next_review else 'Never'}")
        
        # End session
        session.end_time = datetime.now()
        self.db.save_study_session(session)
        
        print("\n" + "=" * 50)
        print(f"🎯 Study session completed!")
        print(f"📊 Accuracy: {session.accuracy:.1f}%")
        print(f"⏱️  Duration: {session.duration}")
        
        return session
    
    def get_study_recommendations(self) -> Dict:
        """
        Get personalized study recommendations.
        
        Returns:
            Dictionary with study recommendations
        """
        all_cards = self.db.get_all_flashcards()
        recommendations = self.sr.get_study_recommendations(all_cards)
        
        return recommendations
    
    def show_progress(self):
        """Display learning progress and statistics."""
        progress = self.db.get_learning_progress()
        recommendations = self.get_study_recommendations()
        
        print("\n📈 Your Learning Progress")
        print("=" * 40)
        print(f"Total Cards: {progress.total_cards}")
        print(f"Mastered Cards: {progress.mastered_cards}")
        print(f"Mastery Rate: {progress.mastery_percentage:.1f}%")
        print(f"Total Study Time: {progress.total_study_time}")
        print(f"Study Sessions: {progress.total_sessions}")
        
        print(f"\n📋 Study Recommendations")
        print("=" * 40)
        print(f"Cards Due: {recommendations['due_cards']}")
        print(f"Overdue Cards: {recommendations['overdue_cards']}")
        print(f"New Cards: {recommendations['new_cards']}")
        print(f"Study Priority: {recommendations['study_priority'].upper()}")
    
    def interactive_menu(self):
        """Run the interactive menu system."""
        while True:
            print("\n🤖 AI Teacher - Main Menu")
            print("=" * 30)
            print("1. Create Flashcard")
            print("2. Add Note")
            print("3. Generate Flashcards from Text")
            print("4. Start Study Session")
            print("5. View Progress")
            print("6. View All Flashcards")
            print("7. View All Notes")
            print("8. Exit")
            
            choice = input("\nChoose an option (1-8): ").strip()
            
            if choice == "1":
                self._create_flashcard_interactive()
            elif choice == "2":
                self._add_note_interactive()
            elif choice == "3":
                self._generate_flashcards_interactive()
            elif choice == "4":
                self.start_study_session()
            elif choice == "5":
                self.show_progress()
            elif choice == "6":
                self._view_flashcards()
            elif choice == "7":
                self._view_notes()
            elif choice == "8":
                print("👋 Thanks for using AI Teacher! Keep learning!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def _create_flashcard_interactive(self):
        """Interactive flashcard creation."""
        term = input("Enter the term: ").strip()
        if not term:
            print("❌ Term cannot be empty")
            return
        
        definition = input("Enter the definition: ").strip()
        if not definition:
            print("❌ Definition cannot be empty")
            return
        
        category = input("Enter category (optional, press Enter for 'general'): ").strip()
        if not category:
            category = "general"
        
        self.create_flashcard(term, definition, category)
    
    def _add_note_interactive(self):
        """Interactive note creation."""
        title = input("Enter note title: ").strip()
        if not title:
            print("❌ Title cannot be empty")
            return
        
        print("Enter note content (press Enter twice when done):")
        content_lines = []
        while True:
            line = input()
            if line == "" and content_lines and content_lines[-1] == "":
                break
            content_lines.append(line)
        
        content = "\n".join(content_lines[:-1])  # Remove last empty line
        if not content.strip():
            print("❌ Content cannot be empty")
            return
        
        category = input("Enter category (optional, press Enter for 'general'): ").strip()
        if not category:
            category = "general"
        
        self.add_note(title, content, category)
    
    def _generate_flashcards_interactive(self):
        """Interactive flashcard generation from text."""
        print("Enter text to generate flashcards from (press Enter twice when done):")
        text_lines = []
        while True:
            line = input()
            if line == "" and text_lines and text_lines[-1] == "":
                break
            text_lines.append(line)
        
        text = "\n".join(text_lines[:-1])  # Remove last empty line
        if not text.strip():
            print("❌ Text cannot be empty")
            return
        
        category = input("Enter category (optional, press Enter for 'general'): ").strip()
        if not category:
            category = "general"
        
        self.create_flashcards_from_text(text, category)
    
    def _view_flashcards(self):
        """View all flashcards."""
        flashcards = self.db.get_all_flashcards()
        
        if not flashcards:
            print("📝 No flashcards found. Create some to get started!")
            return
        
        print(f"\n📚 All Flashcards ({len(flashcards)})")
        print("=" * 50)
        
        for i, card in enumerate(flashcards, 1):
            print(f"\n{i}. {card.term}")
            print(f"   Definition: {card.definition}")
            print(f"   Category: {card.category}")
            print(f"   Reviews: {card.review_count}")
            print(f"   Difficulty: {card.difficulty:.1f}")
            if card.next_review:
                print(f"   Next Review: {card.next_review.strftime('%Y-%m-%d')}")
    
    def _view_notes(self):
        """View all notes."""
        notes = self.db.get_all_notes()
        
        if not notes:
            print("📝 No notes found. Add some to get started!")
            return
        
        print(f"\n📝 All Notes ({len(notes)})")
        print("=" * 50)
        
        for i, note in enumerate(notes, 1):
            print(f"\n{i}. {note.title}")
            print(f"   Category: {note.category}")
            print(f"   Created: {note.created_at.strftime('%Y-%m-%d')}")
            if note.summary:
                print(f"   Summary: {note.summary}")
            print(f"   Content: {note.content[:100]}{'...' if len(note.content) > 100 else ''}")


def main():
    """Main entry point for the AI Teacher application."""
    print("🚀 Starting AI Teacher...")
    
    # Initialize the application
    ai_teacher = AITeacher()
    
    # Run the interactive menu
    ai_teacher.interactive_menu()


if __name__ == "__main__":
    main()
