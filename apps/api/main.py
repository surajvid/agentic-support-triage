from __future__ import annotations

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from prometheus_client import Counter, Histogram, generate_latest

from .database import engine, get_db
from .models import Base
from . import crud, schemas

from agents.triage.runner import run_triage_agent


# ---- Create tables (MVP) ----
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Agentic Support Triage API")


# ---- Metrics ----
tickets_total = Counter("tickets_total", "Total tickets received")
agent_runs_total = Counter("agent_runs_total", "Total agent runs executed")
auto_send_total = Counter("agent_auto_send_total", "Agent decisions: auto_send")
human_review_total = Counter("agent_human_review_total", "Agent decisions: human_review")
escalate_total = Counter("agent_escalate_total", "Agent decisions: escalate")

agent_latency_ms = Histogram("agent_latency_ms", "Agent end-to-end latency (ms)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/tickets", response_model=schemas.TicketProcessResponse)
def create_and_process_ticket(payload: schemas.TicketCreate, db: Session = Depends(get_db)):
    """
    1) Store the raw ticket
    2) Run triage pipeline (PII -> classify -> draft -> decide)
    3) Store agent_run
    4) Create review if not auto_send
    5) Return everything
    """
    tickets_total.inc()

    # 1) Save ticket
    ticket = crud.create_ticket(
        db=db,
        subject=payload.subject,
        body=payload.body,
        customer_email=str(payload.customer_email) if payload.customer_email else None,
        channel=payload.channel,
    )

    # 2) Run agent pipeline
    with agent_latency_ms.time():
        output = run_triage_agent(
            subject=payload.subject,
            body=payload.body,
            customer_email=str(payload.customer_email) if payload.customer_email else None,
            channel=payload.channel,
        )

    agent_runs_total.inc()

    # Extract fields we store in agent_runs table
    cls = output.get("classification", {})
    dr = output.get("draft", {})
    dec = output.get("decision", {})

    draft_text_for_db = None
    if isinstance(dr, dict):
        # Store a compact reply body in DB for quick review
        subj = dr.get("subject", "")
        body = dr.get("body", "")
        draft_text_for_db = f"SUBJECT: {subj}\n\n{body}".strip()

    # 3) Save agent run
    agent_run = crud.create_agent_run(
        db=db,
        ticket_id=ticket.id,
        intent=cls.get("intent"),
        priority=cls.get("priority"),
        confidence=cls.get("confidence"),
        draft_reply=draft_text_for_db,
        decision=dec.get("decision"),
    )

    # 4) Metrics by decision
    decision_value = (dec.get("decision") or "").lower()
    review_obj = None

    if decision_value == "auto_send":
        auto_send_total.inc()
        # In real integration, you would send email here.
        # For now we only record the decision.
    else:
        if decision_value == "human_review":
            human_review_total.inc()
        elif decision_value == "escalate":
            escalate_total.inc()
        else:
            human_review_total.inc()

        review_obj = crud.create_review(
            db=db,
            agent_run_id=agent_run.id,
            status="pending",
        )

    return {
        "ticket": ticket,
        "agent_run": agent_run,
        "review": review_obj,
        "pipeline_output": output,
    }


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    ticket = crud.get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    agent_run = crud.get_latest_agent_run(db, ticket_id)
    return {
        "ticket": schemas.TicketOut.model_validate(ticket),
        "latest_agent_run": schemas.AgentRunOut.model_validate(agent_run) if agent_run else None,
    }


@app.get("/reviews", response_model=list[schemas.ReviewOut])
def list_reviews(db: Session = Depends(get_db), limit: int = 50):
    return crud.list_pending_reviews(db, limit=limit)


@app.post("/reviews/{review_id}/approve", response_model=schemas.ReviewOut)
def approve_review(review_id: int, final_reply: str, db: Session = Depends(get_db)):
    review = crud.get_review(db, review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    review.status = "approved"
    review.final_reply = final_reply
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


@app.post("/reviews/{review_id}/reject", response_model=schemas.ReviewOut)
def reject_review(review_id: int, reviewer_notes: str, db: Session = Depends(get_db)):
    review = crud.get_review(db, review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    review.status = "rejected"
    review.reviewer_notes = reviewer_notes
    db.add(review)
    db.commit()
    db.refresh(review)
    return review

