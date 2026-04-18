"""
Banking Policy Intelligence Platform - Analytics & Feedback System

SQLite-based analytics for:
- Query logging (every question, answer, sources, response time)
- User feedback (thumbs up/down)
- Usage patterns (most asked topics, low confidence answers)
- Gap identification (unanswered questions)
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class AnalyticsEngine:
    """Tracks queries, feedback, and usage patterns"""

    def __init__(self, db_path: str = './analytics.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                intent_category TEXT,
                answer TEXT,
                sources_json TEXT,
                num_sources INTEGER,
                response_time_ms INTEGER,
                confidence TEXT,
                retrieval_method TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT,
                FOREIGN KEY (query_id) REFERENCES queries(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                num_queries INTEGER DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()

    def log_query(
        self,
        question: str,
        answer: str,
        sources: List[Dict],
        intent_category: str = 'GENERAL',
        response_time_ms: int = 0,
        confidence: str = 'Medium',
        retrieval_method: str = 'Hybrid',
    ) -> int:
        """Log a query and return the query ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO queries (timestamp, question, intent_category, answer,
                               sources_json, num_sources, response_time_ms,
                               confidence, retrieval_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            question,
            intent_category,
            answer,
            json.dumps(sources, default=str),
            len(sources),
            response_time_ms,
            confidence,
            retrieval_method,
        ))

        query_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return query_id

    def log_feedback(self, query_id: int, rating: int, comment: str = '') -> None:
        """
        Log user feedback for a query.
        rating: 1 (thumbs up) or -1 (thumbs down)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO feedback (query_id, timestamp, rating, comment)
            VALUES (?, ?, ?, ?)
        ''', (query_id, datetime.now().isoformat(), rating, comment))

        conn.commit()
        conn.close()

    def get_stats(self) -> Dict:
        """Get overall analytics statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total queries
        cursor.execute('SELECT COUNT(*) FROM queries')
        total_queries = cursor.fetchone()[0]

        # Average response time
        cursor.execute('SELECT AVG(response_time_ms) FROM queries WHERE response_time_ms > 0')
        avg_response_time = cursor.fetchone()[0] or 0

        # Feedback stats
        cursor.execute('SELECT COUNT(*) FROM feedback WHERE rating = 1')
        positive_feedback = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM feedback WHERE rating = -1')
        negative_feedback = cursor.fetchone()[0]

        # Most common intent categories
        cursor.execute('''
            SELECT intent_category, COUNT(*) as count
            FROM queries
            GROUP BY intent_category
            ORDER BY count DESC
            LIMIT 5
        ''')
        top_intents = cursor.fetchall()

        # Recent queries
        cursor.execute('''
            SELECT question, intent_category, confidence, timestamp
            FROM queries
            ORDER BY id DESC
            LIMIT 10
        ''')
        recent_queries = cursor.fetchall()

        conn.close()

        satisfaction_rate = 0
        total_feedback = positive_feedback + negative_feedback
        if total_feedback > 0:
            satisfaction_rate = round(positive_feedback / total_feedback * 100, 1)

        return {
            'total_queries': total_queries,
            'avg_response_time_ms': round(avg_response_time, 0),
            'positive_feedback': positive_feedback,
            'negative_feedback': negative_feedback,
            'satisfaction_rate': satisfaction_rate,
            'top_intents': [{'category': c, 'count': n} for c, n in top_intents],
            'recent_queries': [
                {
                    'question': q[0],
                    'intent': q[1],
                    'confidence': q[2],
                    'timestamp': q[3],
                }
                for q in recent_queries
            ],
        }

    def get_gap_analysis(self) -> List[Dict]:
        """
        Find questions that got low confidence answers or negative feedback.
        These represent gaps in the knowledge base.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT q.question, q.confidence, q.intent_category, q.timestamp,
                   COALESCE(f.rating, 0) as feedback_rating
            FROM queries q
            LEFT JOIN feedback f ON q.id = f.query_id
            WHERE q.confidence = 'Low' OR f.rating = -1
            ORDER BY q.timestamp DESC
            LIMIT 20
        ''')
        gaps = cursor.fetchall()
        conn.close()

        return [
            {
                'question': g[0],
                'confidence': g[1],
                'intent': g[2],
                'timestamp': g[3],
                'feedback': 'Negative' if g[4] == -1 else 'None',
            }
            for g in gaps
        ]

    def get_topic_distribution(self) -> Dict:
        """Get distribution of questions across banking topics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT intent_category, COUNT(*) as count
            FROM queries
            GROUP BY intent_category
            ORDER BY count DESC
        ''')
        distribution = cursor.fetchall()
        conn.close()

        total = sum(count for _, count in distribution)
        return {
            category: {
                'count': count,
                'percentage': round(count / total * 100, 1) if total > 0 else 0,
            }
            for category, count in distribution
        }
