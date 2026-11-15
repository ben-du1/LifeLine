"""Caller phone numbers (owner: workstream D).

The number comes from the channel: ElevenLabs' `system__caller_id` for voice, the sender handle for iMessage. Both can
be missing or withheld, and SIP trunks sometimes send a URI ("sip:+16095550100@host") instead of a bare number.
"""

import phonenumbers

# What carriers and SIP trunks send instead of a number when the caller ID is blocked.
WITHHELD = {"anonymous", "unknown", "restricted", "private", "unavailable", "withheld", "blocked"}


def to_e164(raw: str | None) -> str | None:
    """A caller's number in E.164 ("+16095551234"), or None when it is missing, withheld or not a phone number.
    Numbers without a country code are read as US numbers."""
    text = (raw or "").strip()
    if text.lower().startswith(("sip:", "sips:", "tel:")):
        text = text.split(":", 1)[1].split("@", 1)[0].split(";", 1)[0]
    if not text or text.lower() in WITHHELD:
        return None
    try:
        number = phonenumbers.parse(text, "US")
    except phonenumbers.NumberParseException:
        return None
    # The 555 area code is fictional, so phonenumbers calls it invalid, but the project's test callers use
    # "+1 555 ..." numbers on purpose (CLAUDE.MD): accept it when the length is right.
    fictional = number.country_code == 1 and str(number.national_number).startswith("555")
    if not (phonenumbers.is_valid_number(number) or (fictional and phonenumbers.is_possible_number(number))):
        return None
    return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)
