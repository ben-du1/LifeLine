"""Hand-off to standard emergency services for addresses outside the Matterport coverage (contract; owner: B).

This build connects nothing: it records what would happen so it can be shown and explained.
"""

import time

TARGET = "standard 911 (PSAP)"


def transfer_to_emergency_services(session, reason: str) -> dict:
    """Mark the call as handed off and return the transfer record. Phase 0 stub: status and record only."""
    transfer = {"reason": reason, "ts": time.time(), "target": TARGET, "stub": True}
    session.status = "transferred"
    session.transfer = transfer
    return transfer
