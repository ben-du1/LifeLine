import test from "node:test";
import assert from "node:assert/strict";
import { once } from "node:events";
import { createGateway } from "./gateway.mjs";

test("gateway exposes only authenticated voice tools and forwards the transcript unchanged", async () => {
  const forwarded = [];
  const server = createGateway({ token: "test-secret", fetchImpl: async (url, options) => {
    forwarded.push({ url: String(url), options });
    return new Response('{"say":"Are there windows?"}', { status: 200 });
  } });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const origin = "http://127.0.0.1:" + server.address().port;
  try {
    for (const path of ["/api/sessions", "/api/calls", "/buildings/secret", "/api/voice/reply?x=1"]) {
      const response = await fetch(origin + path, { method: "POST" });
      assert.equal(response.status, 404);
    }
    assert.equal((await fetch(origin + "/api/voice/reply", { method: "POST" })).status, 401);
    const body = JSON.stringify({ conversation_id: "c", history: "yes, actually upstairs" });
    const response = await fetch(origin + "/api/voice/reply", {
      method: "POST", headers: { Authorization: "Bearer test-secret" }, body,
    });
    assert.equal(response.status, 200);
    assert.equal((await response.json()).say, "Are there windows?");
    assert.equal(forwarded.length, 1);
    assert.equal(forwarded[0].options.body, body);
    assert.equal(forwarded[0].url, "http://127.0.0.1:8000/api/voice/reply");
    assert.equal((await fetch(origin + "/api/voice/reply", {
      method: "POST", headers: { Authorization: "Bearer test-secret" }, body: "{",
    })).status, 400);
  } finally {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
});
