"""
database.py — SQLAlchemy setup for the Shafaco backend.

Uses a local SQLite database (reports.db) to store audio analysis reports.
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

DATABASE_URL = "sqlite:///./reports.db"

engine = create_engine(
    DATABASE_URL,
    # Required for SQLite when used with multiple threads (e.g. Uvicorn workers)
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Model
# ---------------------------------------------------------------------------


class Report(Base):
    """Stores a single audio-analysis report."""

    __tablename__ = "reports"

    id: int = Column(Integer, primary_key=True, index=True, autoincrement=True)
    engineer_name: str = Column(String(255), default="غير محدد", nullable=False)
    raw_text: str = Column(Text, nullable=True)
    equipment: str = Column(String(255), nullable=True)
    action_taken: str = Column(Text, nullable=True)
    status: str = Column(String(50), nullable=True)        # طبيعي / يحتاج متابعة / عطل حرج
    audio_path: str = Column(String(512), nullable=True)   # Relative path or URL
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        """Serialize the model instance to a plain dictionary."""
        return {
            "id": self.id,
            "engineer_name": self.engineer_name,
            "raw_text": self.raw_text,
            "equipment": self.equipment,
            "action_taken": self.action_taken,
            "status": self.status,
            "audio_path": self.audio_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create all tables in the database (idempotent)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    FastAPI dependency that yields a database session and ensures it is
    properly closed after the request completes.
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
