from services.pii.redact import redact_pii


def test_redact_email_and_phone():
    text = "Contact me at test.user@gmail.com or 9876543210"
    redacted, findings = redact_pii(text)

    assert "[REDACTED:EMAIL]" in redacted
    assert "[REDACTED:PHONE]" in redacted
    assert any(f["type"] == "EMAIL" for f in findings)
    assert any(f["type"] == "PHONE" for f in findings)
