"""
Database table definitions for the Support Triage System.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from .database import Base


# --------------------------------------------------------
# TABLE 1: Tickets
# --------------------------------------------------------
class Ticket(Base):
    """
    Stores incoming customer tickets.
    """

    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String(255))
    body = Column(Text)
    customer_email = Column(String(255))
    channel = Column(String(50))  # email / chat / etc
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to agent runs
    agent_runs = relationship("AgentRun", back_populates="ticket")


# --------------------------------------------------------
# TABLE 2: Agent Runs
# --------------------------------------------------------
class AgentRun(Base):
    """
    Stores AI agent results for each ticket.
    """

    __tablename__ = "agent_runs"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"))

    intent = Column(String(100))
    priority = Column(String(10))
    confidence = Column(Float)

    draft_reply = Column(Text)
    decision = Column(String(50))  # auto_send / human_review

    created_at = Column(DateTime, default=datetime.utcnow)

    ticket = relationship("Ticket", back_populates="agent_runs")
    reviews = relationship("Review", back_populates="agent_run") 


# --------------------------------------------------------
# TABLE 3: Reviews
# --------------------------------------------------------
class Review(Base):
    """
    Stores human review decisions.
    """

    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    agent_run_id = Column(Integer, ForeignKey("agent_runs.id"))

    status = Column(String(50))  # pending / approved / rejected
    reviewer_notes = Column(Text)
    final_reply = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    agent_run = relationship("AgentRun", back_populates="reviews") 
