"""
Agent Runner (Step 5.9)

This module orchestrates the entire agent pipeline:

1) Redact PII (privacy protection)
2) Classify ticket (intent + priority + confidence)
3) Draft reply grounded in KB
4) Decide action (auto_send vs human_review vs escalate)

This is a simple orchestrator for MVP.
Later we can replace this with a LangGraph graph without changing the API layer.
"""

from __future__ import annotations

import time
from typing import Dict, Any

from services.pii.redact import redact_pii
from agents.triage.classify import classify_ticket
from agents.triage.draft_reply import draft_reply
from agents.triage.decision import decide_action


def run_triage_agent(
    subject: str,
    body: str,
    customer_email: str | None = None,
    channel: str = "api",
) -> Dict[str, Any]:
    """
    Runs the full triage pipeline and returns a structured result.

    Args:
        subject: ticket subject/title
        body: ticket message body
        customer_email: optional (do not send to LLM)
        channel: source of ticket (email/chat/api)

    Returns:
        dict containing:
        - redacted_text
        - pii_findings
        - classification (intent, priority, confidence, reasoning)
        - draft (subject, body, citations, needs_clarification)
        - decision (decision, reason, auto_send_allowed)
        - latency_ms
    """

    start = time.time()

    # -----------------------------
    # 1) Preprocess: redact PII
    # -----------------------------
    # We redact subject + body together to avoid leaking any PII to LLM.
    combined = f"SUBJECT: {subject}\nBODY: {body}"
    redacted_combined, pii_findings = redact_pii(combined, keep_values_in_findings=False)

    # Split back into subject/body (simple approach)
    # Note: This depends on our format "SUBJECT: ...\nBODY: ..."
    # For MVP this is fine.
    redacted_subject = subject
    redacted_body = body
    if "SUBJECT:" in redacted_combined and "\nBODY:" in redacted_combined:
        try:
            redacted_subject = redacted_combined.split("SUBJECT:", 1)[1].split("\nBODY:", 1)[0].strip()
            redacted_body = redacted_combined.split("\nBODY:", 1)[1].strip()
        except Exception:
            # fallback to original if parsing fails
            redacted_subject = subject
            redacted_body = body

    # -----------------------------
    # 2) Classification (LLM)
    # -----------------------------
    classification = classify_ticket(
        ticket_subject=redacted_subject,
        ticket_body=redacted_body
    )

    # -----------------------------
    # 3) Draft reply grounded in KB
    # -----------------------------
    draft = draft_reply(
        ticket_subject=redacted_subject,
        ticket_body=redacted_body,
        classification=classification,
        top_k=5
    )

    # -----------------------------
    # 4) Decision policy (deterministic)
    # -----------------------------
    decision = decide_action(
        classification=classification,
        draft=draft
    )

    latency_ms = int((time.time() - start) * 1000)

    return {
        "ticket": {
            "subject": subject,
            "body": body,
            "customer_email": customer_email,
            "channel": channel,
        },
        "redacted": {
            "subject": redacted_subject,
            "body": redacted_body,
            "pii_findings": pii_findings,
        },
        "classification": classification.model_dump(),
        "draft": draft.model_dump(),
        "decision": decision.model_dump(),
        "latency_ms": latency_ms,
    }


# Manual test
if __name__ == "__main__":
    result = run_triage_agent(
        subject="Refund request for my order",
        body="Hi, I purchased yesterday. I want a refund. My email is john.doe@gmail.com"
    )
    from pprint import pprint
    pprint(result)
