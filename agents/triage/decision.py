"""
Decision Policy Node

This node decides what to do with the drafted reply:
- auto_send: safe to send automatically
- human_review: needs human approval
- escalate: very high risk / priority

This is deterministic rule-based logic (not LLM),
which makes the system safer and auditable.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

from agents.triage.schemas import TicketClassification, DraftReply, DecisionResult


def _get_bool_env(key: str, default: str = "false") -> bool:
    return os.getenv(key, default).strip().lower() in ("1", "true", "yes", "y")


def decide_action(
    classification: TicketClassification,
    draft: DraftReply,
) -> DecisionResult:
    """
    Decide what action to take based on:
    - priority
    - confidence
    - intent
    - draft.needs_clarification
    - environment policy settings
    """

    load_dotenv()

    # 1) Read policy settings
    auto_send_enabled = _get_bool_env("AUTO_SEND_ENABLED", "true")
    threshold = float(os.getenv("AUTO_SEND_CONFIDENCE", "0.80"))

    blocked_intents = set(
        x.strip() for x in os.getenv("AUTO_SEND_BLOCKED_INTENTS", "complaint_escalation").split(",") if x.strip()
    )
    blocked_priorities = set(
        x.strip() for x in os.getenv("AUTO_SEND_BLOCKED_PRIORITIES", "P0,P1").split(",") if x.strip()
    )

    # 2) Hard blocks (always human/escalate)
    # If we need clarification, never auto-send.
    if draft.needs_clarification:
        return DecisionResult(
            decision="human_review",
            reason="Draft requires clarification from customer; avoid auto-send.",
            auto_send_allowed=False,
        )

    # P0 always escalate (or human review if you prefer)
    if classification.priority == "P0":
        return DecisionResult(
            decision="escalate",
            reason="Priority P0 detected (security/legal/payment-fraud risk).",
            auto_send_allowed=False,
        )

    # Blocked priorities (P1 usually blocked from auto-send)
    if classification.priority in blocked_priorities:
        return DecisionResult(
            decision="human_review",
            reason=f"Priority {classification.priority} requires human verification.",
            auto_send_allowed=False,
        )

    # Blocked intents (complaints/escalations should be reviewed)
    if classification.intent in blocked_intents:
        return DecisionResult(
            decision="human_review",
            reason=f"Intent '{classification.intent}' is blocked from auto-send.",
            auto_send_allowed=False,
        )

    # 3) If auto-send is disabled globally
    if not auto_send_enabled:
        return DecisionResult(
            decision="human_review",
            reason="Auto-send is disabled by configuration.",
            auto_send_allowed=False,
        )

    # 4) Confidence threshold check
    if classification.confidence < threshold:
        return DecisionResult(
            decision="human_review",
            reason=f"Confidence {classification.confidence:.2f} below threshold {threshold:.2f}.",
            auto_send_allowed=False,
        )

    # 5) Otherwise safe to auto-send
    return DecisionResult(
        decision="auto_send",
        reason="Meets confidence threshold and not blocked by policy.",
        auto_send_allowed=True,
    )


# Manual test
if __name__ == "__main__":
    from agents.triage.schemas import TicketClassification, DraftReply

    classification = TicketClassification(
        intent="shipping_delivery",
        priority="P2",
        confidence=0.85,
        reasoning="Delivery delay, not security related."
    )

    draft = DraftReply(
        subject="Update on your delivery",
        body="Thanks for reaching out...",
        citations=["shipping_delivery.md"],
        needs_clarification=False
    )

    decision = decide_action(classification, draft)
    print(decision.model_dump())
