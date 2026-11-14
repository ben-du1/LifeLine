// Photon iMessage sidecar (Photon Beta / 2.0 REST API, https://api.photon.codes).
//
//   POST /send {to, text}   -> sends an iMessage to `to` (backend calls this when MESSAGING_MODE=photon)
//   GET  /health            -> line mode and the shared-line number callers must text
//   polling listMessages    -> each caller text is forwarded to the backend as POST /api/inbound {from, text}
//
// Why REST and not spectrum-ts: our credentials are Beta (project id "pho_prj_...", key "pho_sk_..."), and spectrum-ts
// is the Stable SDK (it rejects Beta project ids with a 422). Beta auth is a Bearer key; plain fetch is enough.
//
// Shared-line constraint (Photon rule, not ours): a shared iMessage line can only reply in an existing direct
// conversation, to a phone with an active assignment that has texted the line since being assigned. So the caller's
// phone must be assigned in the Photon dashboard and text the line once before we can message it. We learn each
// caller's conversation (space id) and our agent user id from their inbound messages.

import http from "node:http";
import { randomUUID } from "node:crypto";

const PORT = Number(process.env.MESSAGING_PORT || 8787);
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
const PROJECT_ID = process.env.PHOTON_PROJECT_ID;
const API_KEY = process.env.PHOTON_API_KEY;
const API = `https://api.photon.codes/v1/projects/${PROJECT_ID}`;

// 2 s keeps a reply under a few seconds behind the caller without hammering the API.
const POLL_MS = 2000;
// listMessages is eventually consistent, so each poll re-reads this far back and dedupes by message id.
const POLL_OVERLAP_MS = 30_000;
// On startup, scan this far back to rediscover existing caller conversations (we don't forward these).
const HISTORY_MS = 7 * 24 * 3600 * 1000;

if (!PROJECT_ID || !API_KEY) {
  console.error("PHOTON_PROJECT_ID and PHOTON_API_KEY must be set in ../.env");
  process.exit(1);
}

// One direct conversation per caller, keyed by the last 10 digits so "+1 (609) 555-0100" and "6095550100" match.
const conversations = new Map(); // phoneKey -> {spaceId, agentId}
const phoneKey = (phone) => String(phone).replace(/\D/g, "").slice(-10);

async function photon(path, init = {}) {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: { Authorization: `Bearer ${API_KEY}`, "Content-Type": "application/json", ...init.headers },
  });
  const body = await response.text();
  if (!response.ok) throw new Error(`Photon ${init.method || "GET"} ${path} -> ${response.status}: ${body.slice(0, 300)}`);
  return body ? JSON.parse(body) : {};
}

// All received iMessages since `since`, oldest first, following pagination.
async function receivedSince(since) {
  const items = [];
  let pageToken;
  do {
    const query = new URLSearchParams({
      platform: "imessage",
      direction: "received",
      order: "asc",
      createdSince: since.toISOString(),
      pageSize: "100",
    });
    if (pageToken) query.set("pageToken", pageToken);
    const page = await photon(`/messages?${query}`);
    items.push(...page.items);
    pageToken = page.nextPageToken;
  } while (pageToken);
  return items;
}

function remember(message) {
  const from = message.sentBy?.handle;
  if (!from || message.sentBy?.kind !== "human" || message.space?.kind !== "direct") return null;
  conversations.set(phoneKey(from), { spaceId: message.space.id, agentId: message.receivedBy?.id });
  return from;
}

async function sharedLineNumber() {
  const { assignments = [] } = await photon("/platforms/imessage/assignments");
  return assignments[0]?.assignedPhoneNumber;
}

async function send(to, text) {
  const conversation = conversations.get(phoneKey(to));
  if (!conversation?.agentId) {
    const line = await sharedLineNumber().catch(() => undefined);
    const error = new Error(
      `No iMessage conversation with ${to} yet. Photon's shared line can only reply: assign this phone in the Photon ` +
        `dashboard and have it text ${line || "the project's shared line"} first.`,
    );
    error.status = 409;
    throw error;
  }
  await photon("/messages", {
    method: "POST",
    headers: { "Idempotency-Key": randomUUID() },
    body: JSON.stringify({
      spaceId: conversation.spaceId,
      senderId: conversation.agentId,
      content: { type: "text", format: "plain", text },
    }),
  });
}

async function forward(from, text) {
  const response = await fetch(`${BACKEND_URL}/api/inbound`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ from, text }),
  });
  if (!response.ok) console.warn(`backend rejected reply from ${from}: ${response.status}`);
}

// Forward new caller texts to the backend, which matches them to a call by phone number.
async function pollForever(startedAt) {
  const seen = new Set();
  let cursor = startedAt;
  for (;;) {
    try {
      const since = new Date(Math.max(startedAt.getTime(), cursor.getTime() - POLL_OVERLAP_MS));
      for (const message of await receivedSince(since)) {
        if (seen.has(message.id)) continue;
        seen.add(message.id);
        cursor = new Date(Math.max(cursor.getTime(), Date.parse(message.createdAt)));
        const from = remember(message);
        if (!from || message.content?.type !== "text") continue;
        await forward(from, message.content.text);
      }
    } catch (err) {
      console.error("polling Photon failed:", err.message || err);
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
  }
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      try {
        resolve(JSON.parse(body || "{}"));
      } catch (err) {
        reject(err);
      }
    });
  });
}

function reply(res, status, payload) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(payload));
}

const startedAt = new Date();
const history = await receivedSince(new Date(startedAt.getTime() - HISTORY_MS));
history.forEach(remember);
const { mode } = await photon("/platforms/imessage");

http
  .createServer(async (req, res) => {
    if (req.method === "GET" && req.url === "/health") {
      return reply(res, 200, { ok: true, mode, line: await sharedLineNumber().catch(() => null), callers: conversations.size });
    }
    if (req.method !== "POST" || req.url !== "/send") return reply(res, 404, { error: "not found" });
    try {
      const { to, text } = await readJson(req);
      if (!to || !text) return reply(res, 400, { error: "to and text are required" });
      await send(to, text);
      reply(res, 200, { ok: true });
    } catch (err) {
      console.error("send failed:", err.message || err);
      reply(res, err.status || 502, { error: String(err?.message || err) });
    }
  })
  .listen(PORT, () =>
    console.log(`messaging sidecar on :${PORT} (iMessage ${mode} line, ${conversations.size} known callers), forwarding replies to ${BACKEND_URL}`),
  );

pollForever(startedAt);
