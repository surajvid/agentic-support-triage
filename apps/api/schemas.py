from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Any, Dict, List
from datetime import datetime


class TicketCreate(BaseModel):
    subject: str = Field(..., min_length=1, max_length=255)
    body: str = Field(..., min_length=1)
    customer_email: Optional[EmailStr] = None
    channel: str = Field(default="api", max_length=50)


class TicketOut(BaseModel):
    id: int
    subject: str
    body: str
    customer_email: Optional[str]
    channel: str
    created_at: datetime

    class Config:
        from_attributes = True


class AgentRunOut(BaseModel):
    id: int
    ticket_id: int
    intent: Optional[str]
    priority: Optional[str]
    confidence: Optional[float]
    draft_reply: Optional[str]
    decision: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ReviewOut(BaseModel):
    id: int
    agent_run_id: int
    status: str
    reviewer_notes: Optional[str] = None
    final_reply: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class TicketProcessResponse(BaseModel):
    ticket: TicketOut
    agent_run: AgentRunOut
    review: Optional[ReviewOut] = None
    pipeline_output: Dict[str, Any]
