"""Caller phone numbers (contract; owner: workstream D)."""


def to_e164(raw: str | None) -> str | None:
    """A caller's number in E.164 ("+16095551234"), or None when it is missing, withheld or not a phone number.
    Phase 0 stub: today's last-10-digits rule, as "+1" + digits."""
    digits = "".join(ch for ch in (raw or "") if ch.isdigit())[-10:]
    return f"+1{digits}" if len(digits) == 10 else None
