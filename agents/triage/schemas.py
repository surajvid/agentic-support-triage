"""
Pydantic schemas define the structured format we EXPECT from the LLM.

Why we do this:
- The LLM sometimes outputs messy text.
- We want strict JSON so our agent is reliable.
- If validation fails, we can retry safely.
"""

from pydantic import BaseModel, Field
from typing import Literal


Intent = Literal[
    "billing_refund",
    "technical_issue",
    "account_access",
    "shipping_delivery",
    "product_question",
    "complaint_escalation",
    "unknown",
]

Priority = Literal["P0", "P1", "P2", "P3"]


class TicketClassification(BaseModel):
    """
    Output of the classification step.
    """

    intent: Intent = Field(
        ...,
        description="Primary intent of the ticket."
    )

    priority: Priority = Field(
        ...,
        description="Urgency level: P0 highest, P3 lowest."
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence from 0.0 to 1.0."
    )

    reasoning: str = Field(
        ...,
        description="1-2 lines explaining classification (no PII)."
    )
from typing import List

class DraftReply(BaseModel):
    """
    Output of reply drafting step.
    """

    subject: str = Field(..., description="Email subject line for the reply.")
    body: str = Field(..., description="Customer-facing reply in a professional tone.")
    citations: List[str] = Field(
        default_factory=list,
        description="List of KB sources used (filenames)."
    )
    needs_clarification: bool = Field(
        False,
        description="True if we need more info from customer to proceed safely."
    )
