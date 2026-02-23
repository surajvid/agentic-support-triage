from __future__ import annotations

from sqlalchemy.orm import Session

from . import models


def create_ticket(
    db: Session,
    subject: str,
    body: str,
    customer_email: str | None,
    channel: str,
) -> models.Ticket:
    ticket = models.Ticket(
        subject=subject,
        body=body,
        customer_email=customer_email,
        channel=channel,
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket


def create_agent_run(
    db: Session,
    ticket_id: int,
    intent: str | None,
    priority: str | None,
    confidence: float | None,
    draft_reply: str | None,
    decision: str | None,
) -> models.AgentRun:
    run = models.AgentRun(
        ticket_id=ticket_id,
        intent=intent,
        priority=priority,
        confidence=confidence,
        draft_reply=draft_reply,
        decision=decision,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def create_review(
    db: Session,
    agent_run_id: int,
    status: str = "pending",
    reviewer_notes: str | None = None,
    final_reply: str | None = None,
) -> models.Review:
    review = models.Review(
        agent_run_id=agent_run_id,
        status=status,
        reviewer_notes=reviewer_notes,
        final_reply=final_reply,
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


def get_ticket(db: Session, ticket_id: int) -> models.Ticket | None:
    return db.query(models.Ticket).filter(models.Ticket.id == ticket_id).first()


def get_latest_agent_run(db: Session, ticket_id: int) -> models.AgentRun | None:
    return (
        db.query(models.AgentRun)
        .filter(models.AgentRun.ticket_id == ticket_id)
        .order_by(models.AgentRun.created_at.desc())
        .first()
    )


def list_pending_reviews(db: Session, limit: int = 50) -> list[models.Review]:
    return (
        db.query(models.Review)
        .filter(models.Review.status == "pending")
        .order_by(models.Review.created_at.asc())
        .limit(limit)
        .all()
    )


def get_review(db: Session, review_id: int) -> models.Review | None:
    return db.query(models.Review).filter(models.Review.id == review_id).first()
