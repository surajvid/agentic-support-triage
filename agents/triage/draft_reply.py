"""
Draft Reply Node

This module generates a customer reply grounded in KB snippets.

Inputs:
- ticket subject/body (already redacted)
- classification (intent/priority)
- retrieved KB chunks (top_k)
Outputs:
- DraftReply (subject, body, citations, needs_clarification)

Key rule:
- NEVER invent policy. If KB doesn't contain answer, ask a follow-up question.
"""

from __future__ import annotations

import json
from typing import List, Dict, Any

from tenacity import retry, stop_after_attempt, wait_exponential

from services.llm.client import get_chat_model
from services.kb.retrieve import retrieve
from agents.triage.schemas import TicketClassification, DraftReply


def _format_kb_snippets(hits: List[Dict[str, Any]]) -> str:
    """
    Convert retrieved KB hits into a readable block for the LLM prompt.
    """
    lines = []
    for i, h in enumerate(hits, 1):
        src = h.get("source", "unknown_source")
        chunk_id = h.get("chunk_id", "?")
        text = h.get("text", "")
        lines.append(f"[KB {i}] SOURCE={src} CHUNK={chunk_id}\n{text}")
    return "\n\n".join(lines)


def _build_prompt(
    ticket_subject: str,
    ticket_body: str,
    classification: TicketClassification,
    kb_hits: List[Dict[str, Any]],
) -> str:
    """
    Build a strict prompt that forces grounded responses.
    """
    kb_block = _format_kb_snippets(kb_hits)

    return f"""
You are a customer support agent. Draft a helpful reply.

STRICT RULES:
1) Use ONLY the KB snippets below for policy/steps.
2) If KB is insufficient or unclear, ask 1-2 clarification questions.
3) Do NOT mention internal systems, embeddings, vector search, or "KB".
4) Be polite, professional, concise.
5) Output MUST be valid JSON matching schema:
   {{
     "subject": string,
     "body": string,
     "citations": [string],
     "needs_clarification": boolean
   }}

Ticket context:
- Intent: {classification.intent}
- Priority: {classification.priority}
- Confidence: {classification.confidence}

Customer message:
SUBJECT: {ticket_subject}
BODY: {ticket_body}

KB snippets (your only source of truth):
{kb_block}

Now return ONLY JSON.
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def draft_reply(
    ticket_subject: str,
    ticket_body: str,
    classification: TicketClassification,
    top_k: int = 5,
) -> DraftReply:
    """
    Generates a draft reply grounded in retrieved KB snippets.

    Steps:
    1) Build retrieval query
    2) Retrieve KB hits
    3) Build prompt
    4) Call LLM
    5) Parse JSON and validate via Pydantic
    """

    # Retrieval query: intent + customer text gives better hits
    query = f"{classification.intent}. {ticket_subject}\n{ticket_body}"

    kb_hits = retrieve(query, top_k=top_k)

    llm = get_chat_model()
    prompt = _build_prompt(ticket_subject, ticket_body, classification, kb_hits)

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Remove accidental markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    data = json.loads(raw)

    # If citations not provided, auto-fill from KB sources used
    if "citations" not in data or not data["citations"]:
        data["citations"] = sorted({h.get("source", "unknown") for h in kb_hits})

    return DraftReply(**data)


# Manual test
if __name__ == "__main__":
    # Example classification
    classification = TicketClassification(
        intent="shipping_delivery",
        priority="P2",
        confidence=0.82,
        reasoning="Delivery delay; user not blocked, not security issue."
    )

    subject = "My delivery is delayed"
    body = "Hi, my order has not arrived for 6 days. Can you check and advise?"

    result = draft_reply(subject, body, classification)
    print(result.model_dump())
