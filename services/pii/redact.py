"""
PII Redaction Service

Goal:
- Remove or mask sensitive customer data before sending text to LLMs
- Keep a structured record of what was redacted (for auditing/debugging)

This module is used in the agent's preprocess step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class PiiFinding:
    """
    Represents one redacted item.
    - type: what kind of PII it was
    - value: the original detected value (optional in prod; keep for dev)
    - start/end: where it occurred in the original text (best-effort)
    """
    type: str
    value: str
    start: int
    end: int


# -----------------------------
# Regex patterns (MVP)
# -----------------------------
# NOTE: Regex can never be perfect, but these cover common cases well enough for MVP.
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[- ]?)?(?:\d{10}|\d{3}[- ]\d{3}[- ]\d{4})\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# Indian IDs (optional but helpful for your region)
AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)

# (Optional) Basic IP address
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


# -----------------------------
# Helper to replace matches safely
# -----------------------------
def _mask(match: re.Match, label: str) -> str:
    """
    Returns a consistent replacement string for a found PII match.
    Example: [REDACTED:EMAIL]
    """
    return f"[REDACTED:{label}]"


def redact_pii(
    text: str,
    keep_values_in_findings: bool = False,
) -> Tuple[str, List[Dict]]:
    """
    Main redaction function.

    Args:
        text: input customer message
        keep_values_in_findings: if False, do not store original values in findings

    Returns:
        redacted_text: text with PII masked
        findings: list of dicts describing what was found and where
    """

    if not text:
        return text, []

    findings: List[PiiFinding] = []

    # We apply patterns in an order: credit cards first (high risk), then IDs, then email/phone, etc.
    patterns = [
        ("CREDIT_CARD", CREDIT_CARD_RE),
        ("AADHAAR", AADHAAR_RE),
        ("PAN", PAN_RE),
        ("EMAIL", EMAIL_RE),
        ("PHONE", PHONE_RE),
        ("IP", IP_RE),
    ]

    # To capture positions, we first find all matches for each pattern on the ORIGINAL text.
    # Then we replace iteratively. Note: positions in findings refer to original text (best effort).
    for label, regex in patterns:
        for m in regex.finditer(text):
            value = m.group(0)
            findings.append(
                PiiFinding(
                    type=label,
                    value=value if keep_values_in_findings else "",
                    start=m.start(),
                    end=m.end(),
                )
            )

        # Replace matches in the working text
        text = regex.sub(lambda m: _mask(m, label), text)

    # Convert findings to serializable dicts for DB logging
    return text, [asdict(f) for f in findings]


# -----------------------------
# Simple manual test (optional)
# -----------------------------
if __name__ == "__main__":
    sample = (
        "Hi, my email is john.doe@gmail.com and phone is +91 9876543210. "
        "My Aadhaar is 1234 5678 9012 and card 4111 1111 1111 1111."
    )
    redacted, findings = redact_pii(sample, keep_values_in_findings=True)
    print("Original:", sample)
    print("Redacted:", redacted)
    print("Findings:", findings)
