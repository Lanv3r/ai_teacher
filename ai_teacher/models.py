"""
Data models for the AI Teacher application.
This module defines the core data structures used throughout the application.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json


@dataclass
class Flashcard:
    """
    Represents a single flashcard with term and definition.
    Includes spaced repetition scheduling information.
    """
    term: str
    definition: str
    created_at: datetime = field(default_factory=datetime.now)
    last_reviewed: Optional[datetime] = None
    review_count: int = 0
    difficulty: float = 2.5  # 1.0 (easy) to 4.0 (hard)
    next_review: Optional[datetime] = None
    category: str = "general"
    
    def to_dict(self) -> Dict:
        """Convert flashcard to dictionary for JSON serialization."""
        return {
            'term': self.term,
            'definition': self.definition,
            'created_at': self.created_at.isoformat(),
            'last_reviewed': self.last_reviewed.isoformat() if self.last_reviewed else None,
            'review_count': self.review_count,
            'difficulty': self.difficulty,
            'next_review': self.next_review.isoformat() if self.next_review else None,
            'category': self.category
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Flashcard':
        """Create flashcard from dictionary."""
        card = cls(
            term=data['term'],
            definition=data['definition'],
            category=data.get('category', 'general')
        )
        card.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('last_reviewed'):
            card.last_reviewed = datetime.fromisoformat(data['last_reviewed'])
        card.review_count = data.get('review_count', 0)
        card.difficulty = data.get('difficulty', 2.5)
        if data.get('next_review'):
            card.next_review = datetime.fromisoformat(data['next_review'])
        return card


@dataclass
class StudySession:
    """
    Represents a study session with performance tracking.
    """
    session_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    cards_studied: int = 0
    correct_answers: int = 0
    incorrect_answers: int = 0
    categories: List[str] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        """Calculate accuracy percentage."""
        if self.cards_studied == 0:
            return 0.0
        return (self.correct_answers / self.cards_studied) * 100
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate session duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class Note:
    """
    Represents a study note with content and metadata.
    """
    title: str
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert note to dictionary for JSON serialization."""
        return {
            'title': self.title,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'category': self.category,
            'tags': self.tags,
            'summary': self.summary
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Note':
        """Create note from dictionary."""
        note = cls(
            title=data['title'],
            content=data['content'],
            category=data.get('category', 'general'),
            tags=data.get('tags', []),
            summary=data.get('summary')
        )
        note.created_at = datetime.fromisoformat(data['created_at'])
        note.updated_at = datetime.fromisoformat(data['updated_at'])
        return note


@dataclass
class LearningProgress:
    """
    Tracks overall learning progress and statistics.
    """
    total_cards: int = 0
    mastered_cards: int = 0  # Cards with difficulty < 2.0
    total_study_time: timedelta = field(default_factory=lambda: timedelta(0))
    total_sessions: int = 0
    streak_days: int = 0
    last_study_date: Optional[datetime] = None
    
    @property
    def mastery_percentage(self) -> float:
        """Calculate mastery percentage."""
        if self.total_cards == 0:
            return 0.0
        return (self.mastered_cards / self.total_cards) * 100
