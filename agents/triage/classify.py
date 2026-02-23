"""
Ticket Classification (Intent + Priority + Confidence)

This module:
- Builds a careful prompt
- Calls the LLM
- Forces it to output JSON that matches our Pydantic schema
- Retries if it fails validation
"""

from __future__ import annotations

import json
from typing import Dict, Any

from tenacity import retry, stop_after_attempt, wait_exponential

from services.llm.client import get_chat_model
from agents.triage.schemas import TicketClassification


# -----------------------------
# Prompt builder
# -----------------------------
def _build_prompt(ticket_subject: str, ticket_body: str) -> str:
    """
    We create a clear instruction prompt.

    Important rules:
    - Output MUST be valid JSON
    - Must match schema keys exactly
    - Reasoning should not contain PII
    """

    return f"""
You are a support ticket triage classifier.

Classify the ticket into:
- intent: one of
  ["billing_refund","technical_issue","account_access","shipping_delivery","product_question","complaint_escalation","unknown"]
- priority: one of ["P0","P1","P2","P3"]
- confidence: number 0.0 to 1.0
- reasoning: 1-2 short lines (no PII)

Priority guidance:
- P0: security/privacy breach, payment fraud, legal threats, account takeover, credit card exposure
- P1: user blocked from core usage (cannot login, app down, payment failed)
- P2: degraded experience (bug with workaround, delivery delay < 7 days, partial issues)
- P3: general questions, feature requests, low urgency

Ticket:
SUBJECT: {ticket_subject}
BODY: {ticket_body}

Return ONLY valid JSON with keys: intent, priority, confidence, reasoning.
"""


# -----------------------------
# LLM call with retry
# -----------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def classify_ticket(ticket_subject: str, ticket_body: str) -> TicketClassification:
    """
    Calls LLM and returns validated TicketClassification.

    Retry policy:
    - Up to 3 attempts
    - Exponential backoff
    Why? Sometimes models output invalid JSON once in a while.
    """

    llm = get_chat_model()
    prompt = _build_prompt(ticket_subject, ticket_body)

    # Call the model
    response = llm.invoke(prompt)
    raw = response.content.strip()

    # If model accidentally wraps JSON in markdown fences, try to remove them
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Parse JSON
    data: Dict[str, Any] = json.loads(raw)

    # Validate with Pydantic schema (raises error if invalid)
    return TicketClassification(**data)


# Simple local test
if __name__ == "__main__":
    sample_subject = "Refund needed for my order"
    sample_body = "I purchased yesterday but want a refund. Order ID 12345."

    result = classify_ticket(sample_subject, sample_body)
    print(result.model_dump())
