"""
Database management for the AI Teacher application.
Handles data persistence using SQLite and JSON files.
"""

import sqlite3
import json
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from .models import Flashcard, Note, StudySession, LearningProgress


class DatabaseManager:
    """
    Manages data persistence for the AI Teacher application.
    Uses SQLite for structured data and JSON for configuration.
    """
    
    def __init__(self, db_path: str = "ai_teacher.db"):
        """Initialize database manager with SQLite database."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create flashcards table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS flashcards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_reviewed TEXT,
                    review_count INTEGER DEFAULT 0,
                    difficulty REAL DEFAULT 2.5,
                    next_review TEXT,
                    category TEXT DEFAULT 'general'
                )
            """)
            
            # Create notes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    tags TEXT,
                    summary TEXT
                )
            """)
            
            # Create study sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS study_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    cards_studied INTEGER DEFAULT 0,
                    correct_answers INTEGER DEFAULT 0,
                    incorrect_answers INTEGER DEFAULT 0,
                    categories TEXT
                )
            """)
            
            conn.commit()
    
    # Flashcard operations
    def save_flashcard(self, flashcard: Flashcard) -> int:
        """Save a flashcard to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO flashcards 
                (term, definition, created_at, last_reviewed, review_count, 
                 difficulty, next_review, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                flashcard.term,
                flashcard.definition,
                flashcard.created_at.isoformat(),
                flashcard.last_reviewed.isoformat() if flashcard.last_reviewed else None,
                flashcard.review_count,
                flashcard.difficulty,
                flashcard.next_review.isoformat() if flashcard.next_review else None,
                flashcard.category
            ))
            return cursor.lastrowid
    
    def get_flashcard(self, card_id: int) -> Optional[Flashcard]:
        """Get a flashcard by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM flashcards WHERE id = ?", (card_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_flashcard(row)
            return None
    
    def get_all_flashcards(self) -> List[Flashcard]:
        """Get all flashcards."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM flashcards ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [self._row_to_flashcard(row) for row in rows]
    
    def get_flashcards_for_review(self) -> List[Flashcard]:
        """Get flashcards that are due for review."""
        now = datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM flashcards 
                WHERE next_review IS NULL OR next_review <= ?
                ORDER BY next_review ASC, created_at ASC
            """, (now.isoformat(),))
            rows = cursor.fetchall()
            return [self._row_to_flashcard(row) for row in rows]
    
    def update_flashcard(self, card_id: int, flashcard: Flashcard):
        """Update a flashcard in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE flashcards 
                SET term = ?, definition = ?, last_reviewed = ?, 
                    review_count = ?, difficulty = ?, next_review = ?, category = ?
                WHERE id = ?
            """, (
                flashcard.term,
                flashcard.definition,
                flashcard.last_reviewed.isoformat() if flashcard.last_reviewed else None,
                flashcard.review_count,
                flashcard.difficulty,
                flashcard.next_review.isoformat() if flashcard.next_review else None,
                flashcard.category,
                card_id
            ))
    
    def delete_flashcard(self, card_id: int):
        """Delete a flashcard from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM flashcards WHERE id = ?", (card_id,))
    
    def _row_to_flashcard(self, row) -> Flashcard:
        """Convert database row to Flashcard object."""
        flashcard = Flashcard(
            term=row[1],
            definition=row[2],
            category=row[8]
        )
        flashcard.created_at = datetime.fromisoformat(row[3])
        if row[4]:
            flashcard.last_reviewed = datetime.fromisoformat(row[4])
        flashcard.review_count = row[5]
        flashcard.difficulty = row[6]
        if row[7]:
            flashcard.next_review = datetime.fromisoformat(row[7])
        return flashcard
    
    # Note operations
    def save_note(self, note: Note) -> int:
        """Save a note to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO notes 
                (title, content, created_at, updated_at, category, tags, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                note.title,
                note.content,
                note.created_at.isoformat(),
                note.updated_at.isoformat(),
                note.category,
                json.dumps(note.tags),
                note.summary
            ))
            return cursor.lastrowid
    
    def get_all_notes(self) -> List[Note]:
        """Get all notes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM notes ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            return [self._row_to_note(row) for row in rows]
    
    def _row_to_note(self, row) -> Note:
        """Convert database row to Note object."""
        note = Note(
            title=row[1],
            content=row[2],
            category=row[5],
            summary=row[7]
        )
        note.created_at = datetime.fromisoformat(row[3])
        note.updated_at = datetime.fromisoformat(row[4])
        if row[6]:
            note.tags = json.loads(row[6])
        return note
    
    # Study session operations
    def save_study_session(self, session: StudySession):
        """Save a study session to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO study_sessions 
                (session_id, start_time, end_time, cards_studied, 
                 correct_answers, incorrect_answers, categories)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.start_time.isoformat(),
                session.end_time.isoformat() if session.end_time else None,
                session.cards_studied,
                session.correct_answers,
                session.incorrect_answers,
                json.dumps(session.categories)
            ))
    
    def get_study_sessions(self, limit: int = 10) -> List[StudySession]:
        """Get recent study sessions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM study_sessions 
                ORDER BY start_time DESC 
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [self._row_to_study_session(row) for row in rows]
    
    def _row_to_study_session(self, row) -> StudySession:
        """Convert database row to StudySession object."""
        session = StudySession(session_id=row[1])
        session.start_time = datetime.fromisoformat(row[2])
        if row[3]:
            session.end_time = datetime.fromisoformat(row[3])
        session.cards_studied = row[4]
        session.correct_answers = row[5]
        session.incorrect_answers = row[6]
        if row[7]:
            session.categories = json.loads(row[7])
        return session
    
    def get_learning_progress(self) -> LearningProgress:
        """Get overall learning progress statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get total cards and mastered cards
            cursor.execute("SELECT COUNT(*) FROM flashcards")
            total_cards = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM flashcards WHERE difficulty < 2.0")
            mastered_cards = cursor.fetchone()[0]
            
            # Get total study time
            cursor.execute("""
                SELECT SUM(
                    CASE 
                        WHEN end_time IS NOT NULL 
                        THEN (julianday(end_time) - julianday(start_time)) * 24 * 60 * 60
                        ELSE 0 
                    END
                ) FROM study_sessions
            """)
            total_seconds = cursor.fetchone()[0] or 0
            total_study_time = timedelta(seconds=total_seconds)
            
            # Get total sessions
            cursor.execute("SELECT COUNT(*) FROM study_sessions")
            total_sessions = cursor.fetchone()[0]
            
            # Get last study date
            cursor.execute("""
                SELECT MAX(start_time) FROM study_sessions 
                WHERE end_time IS NOT NULL
            """)
            last_study = cursor.fetchone()[0]
            last_study_date = datetime.fromisoformat(last_study) if last_study else None
            
            return LearningProgress(
                total_cards=total_cards,
                mastered_cards=mastered_cards,
                total_study_time=total_study_time,
                total_sessions=total_sessions,
                last_study_date=last_study_date
            )
