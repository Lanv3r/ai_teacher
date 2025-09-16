"""
Spaced Repetition Algorithm for the AI Teacher application.
Implements the SM-2 algorithm for optimal learning intervals.
"""

from datetime import datetime, timedelta
from typing import Tuple
import math
from .models import Flashcard


class SpacedRepetition:
    """
    Implements the SM-2 spaced repetition algorithm.
    This algorithm determines when to review flashcards based on performance.
    """
    
    def __init__(self):
        """Initialize the spaced repetition system."""
        # SM-2 algorithm parameters
        self.min_interval = 1  # Minimum interval in days
        self.max_interval = 365  # Maximum interval in days
        self.easy_bonus = 1.3  # Bonus multiplier for easy answers
        self.hard_penalty = 1.2  # Penalty multiplier for hard answers
    
    def calculate_next_review(self, flashcard: Flashcard, quality: int) -> Tuple[datetime, float]:
        """
        Calculate the next review date and updated difficulty based on answer quality.
        
        Args:
            flashcard: The flashcard being reviewed
            quality: Answer quality (0-5 scale)
                    0: Complete blackout
                    1: Incorrect response; correct one remembered
                    2: Incorrect response; where correct one seemed easy to recall
                    3: Correct response after hesitation
                    4: Correct response after a delay
                    5: Perfect response
        
        Returns:
            Tuple of (next_review_date, updated_difficulty)
        """
        if quality < 3:
            # Incorrect answer - reset to minimum interval
            flashcard.review_count = 0
            flashcard.difficulty = max(1.3, flashcard.difficulty - 0.2)
            next_interval = self.min_interval
        else:
            # Correct answer - calculate new interval
            flashcard.review_count += 1
            
            if flashcard.review_count == 1:
                next_interval = 1
            elif flashcard.review_count == 2:
                next_interval = 6
            else:
                # Calculate interval using SM-2 formula
                if quality == 3:
                    # Correct but with hesitation
                    flashcard.difficulty = flashcard.difficulty
                elif quality == 4:
                    # Correct with delay
                    flashcard.difficulty = flashcard.difficulty + 0.1
                elif quality == 5:
                    # Perfect response
                    flashcard.difficulty = flashcard.difficulty + 0.15
                
                # Apply easy bonus or hard penalty
                if quality == 5:
                    next_interval = int(flashcard.difficulty * self.easy_bonus)
                elif quality == 3:
                    next_interval = int(flashcard.difficulty / self.hard_penalty)
                else:
                    next_interval = int(flashcard.difficulty)
                
                # Ensure interval is within bounds
                next_interval = max(self.min_interval, 
                                  min(next_interval, self.max_interval))
        
        # Update flashcard properties
        flashcard.last_reviewed = datetime.now()
        flashcard.next_review = datetime.now() + timedelta(days=next_interval)
        
        return flashcard.next_review, flashcard.difficulty
    
    def get_review_priority(self, flashcard: Flashcard) -> float:
        """
        Calculate review priority for a flashcard.
        Higher priority means the card should be reviewed sooner.
        
        Args:
            flashcard: The flashcard to calculate priority for
            
        Returns:
            Priority score (higher = more urgent)
        """
        if not flashcard.next_review:
            return 1.0  # Never reviewed - highest priority
        
        now = datetime.now()
        time_overdue = (now - flashcard.next_review).total_seconds()
        
        if time_overdue <= 0:
            # Not yet due
            return 0.0
        
        # Calculate priority based on how overdue the card is
        # and its difficulty level
        days_overdue = time_overdue / (24 * 3600)
        priority = days_overdue * (1 + flashcard.difficulty / 4.0)
        
        return priority
    
    def get_due_cards(self, flashcards: list) -> list:
        """
        Get flashcards that are due for review, sorted by priority.
        
        Args:
            flashcards: List of flashcards to check
            
        Returns:
            List of due flashcards sorted by priority (highest first)
        """
        due_cards = []
        
        for card in flashcards:
            if not card.next_review or card.next_review <= datetime.now():
                priority = self.get_review_priority(card)
                due_cards.append((priority, card))
        
        # Sort by priority (highest first)
        due_cards.sort(key=lambda x: x[0], reverse=True)
        
        return [card for _, card in due_cards]
    
    def get_study_recommendations(self, flashcards: list) -> dict:
        """
        Get study recommendations based on current flashcard status.
        
        Args:
            flashcards: List of all flashcards
            
        Returns:
            Dictionary with study recommendations
        """
        total_cards = len(flashcards)
        due_cards = self.get_due_cards(flashcards)
        overdue_cards = [card for card in due_cards 
                        if card.next_review and card.next_review < datetime.now()]
        
        # Calculate statistics
        mastered_cards = len([card for card in flashcards if card.difficulty < 2.0])
        new_cards = len([card for card in flashcards if card.review_count == 0])
        
        recommendations = {
            'total_cards': total_cards,
            'due_cards': len(due_cards),
            'overdue_cards': len(overdue_cards),
            'mastered_cards': mastered_cards,
            'new_cards': new_cards,
            'mastery_percentage': (mastered_cards / total_cards * 100) if total_cards > 0 else 0,
            'study_priority': 'high' if len(overdue_cards) > 5 else 'medium' if len(due_cards) > 0 else 'low'
        }
        
        return recommendations


class StudyPlanner:
    """
    Plans study sessions based on spaced repetition algorithm.
    """
    
    def __init__(self, spaced_repetition: SpacedRepetition):
        """Initialize study planner with spaced repetition system."""
        self.sr = spaced_repetition
    
    def create_study_session(self, flashcards: list, max_cards: int = 20) -> list:
        """
        Create an optimal study session from available flashcards.
        
        Args:
            flashcards: List of all flashcards
            max_cards: Maximum number of cards for the session
            
        Returns:
            List of flashcards to study in this session
        """
        due_cards = self.sr.get_due_cards(flashcards)
        
        # Prioritize overdue cards
        overdue_cards = [card for card in due_cards 
                        if card.next_review and card.next_review < datetime.now()]
        
        # Add new cards if we have space
        new_cards = [card for card in flashcards if card.review_count == 0]
        
        session_cards = []
        
        # Add overdue cards first (up to 60% of session)
        overdue_limit = min(len(overdue_cards), int(max_cards * 0.6))
        session_cards.extend(overdue_cards[:overdue_limit])
        
        # Add remaining due cards
        remaining_slots = max_cards - len(session_cards)
        other_due_cards = [card for card in due_cards if card not in overdue_cards]
        session_cards.extend(other_due_cards[:remaining_slots])
        
        # Add new cards if we still have space
        remaining_slots = max_cards - len(session_cards)
        session_cards.extend(new_cards[:remaining_slots])
        
        return session_cards
    
    def estimate_session_time(self, cards: list) -> timedelta:
        """
        Estimate how long a study session will take.
        
        Args:
            cards: List of flashcards in the session
            
        Returns:
            Estimated session duration
        """
        # Base time per card (in seconds)
        base_time_per_card = 30
        
        # Adjust based on card difficulty
        total_time = 0
        for card in cards:
            time_per_card = base_time_per_card * (1 + card.difficulty / 4.0)
            total_time += time_per_card
        
        return timedelta(seconds=total_time)
