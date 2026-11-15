"""phones.to_e164: the caller's number from caller ID or a sender handle, or None when there isn't one."""

import pytest

from app import phones


@pytest.mark.parametrize("raw, expected", [
    ("+1 (609) 555-0100", "+16095550100"),
    ("609-555-0100", "+16095550100"),
    ("6095550100", "+16095550100"),
    ("1 609 555 0100", "+16095550100"),
    ("+16095550100", "+16095550100"),
    ("+44 20 7183 8750", "+442071838750"),            # a foreign caller keeps their country code
    ("sip:+16095550100@sip.example.com", "+16095550100"),
    ("tel:+1-609-555-0100", "+16095550100"),
    ("+1 555 555 0100", "+15555550100"),             # the project's made-up test numbers
])
def test_formats(raw, expected):
    assert phones.to_e164(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "anonymous", "Anonymous", "unknown", "restricted", "private",
                                 "sip:anonymous@sip.example.com"])
def test_withheld(raw):
    assert phones.to_e164(raw) is None


@pytest.mark.parametrize("raw", ["12345", "not a number", "0000000000", "123-456-7890", "+1 609 555 01000 99"])
def test_invalid(raw):
    assert phones.to_e164(raw) is None
