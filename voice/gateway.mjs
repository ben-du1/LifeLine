// Expose only authenticated voice tools; never proxy dispatcher APIs, assets, or WebSockets.
import { createServer } from "node:http";
import { timingSafeEqual } from "node:crypto";
import { pathToFileURL } from "node:url";

const paths = new Set(["/api/voice/start", "/api/voice/reply", "/api/voice/status"]);

export function createGateway({ token, backendUrl = "http://127.0.0.1:8000", fetchImpl = fetch }) {
  if (!token) throw new Error("VOICE_TOOL_TOKEN is required. Run npm run setup first.");
  const expected = Buffer.from("Bearer " + token);
  return createServer(async (req, res) => {
    const respond = (status, body) => {
      res.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store" });
      res.end(JSON.stringify(body));
    };
    if (req.method !== "POST" || !paths.has(req.url)) return respond(404, { error: "Not found" });
    const supplied = Buffer.from(req.headers.authorization || "");
    if (supplied.length !== expected.length || !timingSafeEqual(supplied, expected)) {
      return respond(401, { error: "Unauthorized" });
    }
    let size = 0;
    const chunks = [];
    try {
      for await (const chunk of req) {
        size += chunk.length;
        if (size > 256000) return respond(413, { error: "Request too large" });
        chunks.push(chunk);
      }
      const body = Buffer.concat(chunks).toString("utf8");
      try { JSON.parse(body); } catch { return respond(400, { error: "Invalid JSON" }); }
      const response = await fetchImpl(new URL(req.url, backendUrl), {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer " + token },
        body,
        signal: AbortSignal.timeout(115000),
        redirect: "error",
      });
      res.writeHead(response.status, { "Content-Type": "application/json", "Cache-Control": "no-store" });
      res.end(await response.text());
    } catch {
      respond(502, { error: "Location backend unavailable. Check the local FastAPI server." });
    }
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const port = Number(process.env.VOICE_GATEWAY_PORT || 8788);
  createGateway({ token: process.env.VOICE_TOOL_TOKEN, backendUrl: process.env.BACKEND_URL })
    .listen(port, "127.0.0.1", () => console.log("Voice gateway listening on http://127.0.0.1:" + port));
}
