"""
Database configuration and session management.

This file:
- Creates the database engine
- Creates session factory
- Provides dependency for FastAPI routes
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found in .env")

# Create database engine
# Engine = connection pool to database
engine = create_engine(DATABASE_URL)

# Session factory
# Session = conversation with database
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for models
Base = declarative_base()


# Dependency for FastAPI routes
def get_db():
    """
    This function provides a database session to API endpoints.
    It ensures the session is closed after request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
