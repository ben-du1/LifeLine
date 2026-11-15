"""Hand-off to standard emergency services for addresses outside the Matterport coverage (owner: workstream B).

This build connects nothing: it records what would happen so it can be shown and explained. Where a real build
connects:
  - voice: an ElevenLabs call transfer (the agent's transfer-to-number tool) or carrier routing of the call to the
    local 911 center (PSAP) for the caller's location;
  - text: Text-to-911, handing the conversation to the PSAP's text gateway with the transcript so far.
The caller is told they are being connected (questions.OUT_OF_AREA_MESSAGE, sent by sessions.py); later texts on a
transferred call are recorded only, like a call whose caller got out.
"""

import logging
import time

log = logging.getLogger(__name__)

TARGET = "standard 911 (PSAP)"
STUB_NOTE = "Transferred to standard 911: address not in coverage. Stub: no real transfer in this build."


def transfer_to_emergency_services(session, reason: str) -> dict:
    """Mark the call as handed off and return the transfer record. No network call (see the module docstring)."""
    transfer = {"reason": reason, "ts": time.time(), "target": TARGET, "stub": True}
    log.info("Session %s handed off to %s (stub): %s", session.id, TARGET, reason)
    session.status = "transferred"
    session.status_detail = f"Transferred to {TARGET}: {reason}"
    session.transfer = transfer
    session.pending, session.queued = None, []
    session.waiting_for_match = False
    session.out_pending = False
    session.messages.append({"role": "system", "text": STUB_NOTE, "ts": transfer["ts"]})
    return transfer
