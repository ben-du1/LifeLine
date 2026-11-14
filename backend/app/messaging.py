"""Outbound messages to the caller.

simulated: nothing leaves the machine; the dashboard shows the message and the dispatcher types the caller's reply.
photon:    POST to the messaging/ sidecar, which sends the iMessage through Photon's REST API.
"""

import logging

import httpx

from . import config

log = logging.getLogger(__name__)


class MessagingError(RuntimeError):
    pass


async def send(phone: str, text: str) -> None:
    if config.MESSAGING_MODE != "photon":
        return
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(f"{config.MESSAGING_SIDECAR_URL}/send", json={"to": phone, "text": text})
    except httpx.HTTPError as exc:
        log.warning("Photon sidecar send failed: %s", exc)
        raise MessagingError(f"Could not send via Photon sidecar: {exc}") from exc
    if response.is_error:
        # The sidecar explains Photon refusals (e.g. the shared line needs the caller to text first); show that text.
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        log.warning("Photon sidecar refused send (%s): %s", response.status_code, detail)
        raise MessagingError(f"Could not send via Photon: {detail}")
