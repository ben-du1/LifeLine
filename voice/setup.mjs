// Provision the real ElevenLabs agent. This command never starts a conversation or places a call.
import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";
import { appendFileSync } from "node:fs";
import { randomBytes } from "node:crypto";

function save(name, value) {
  appendFileSync(new URL("../.env", import.meta.url), "\n" + name + "=" + value + "\n");
  process.env[name] = value;
}

const publicUrl = process.argv[2] || process.env.VOICE_PUBLIC_URL;
if (!publicUrl || new URL(publicUrl).protocol !== "https:") {
  throw new Error("Usage: npm run setup -- https://YOUR-TUNNEL.trycloudflare.com");
}
if (!process.env.ELEVENLABS_API_KEY) throw new Error("Run ElevenLabs onboarding first.");
const client = new ElevenLabsClient();
if (!process.env.VOICE_TOOL_TOKEN) save("VOICE_TOOL_TOKEN", randomBytes(32).toString("hex"));

async function rest(path, body) {
  const response = await fetch("https://api.elevenlabs.io/v1/convai/" + path, {
    method: "POST", headers: { "xi-api-key": process.env.ELEVENLABS_API_KEY, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error("ElevenLabs " + path + " failed (" + response.status + ").");
  return response.json();
}
if (!process.env.ELEVENLABS_VOICE_SECRET_ID) {
  const secret = await rest("secrets", {
    type: "new", name: "HackPrinceton locator " + randomBytes(4).toString("hex"),
    value: "Bearer " + process.env.VOICE_TOOL_TOKEN,
  });
  save("ELEVENLABS_VOICE_SECRET_ID", secret.secret_id);
}
const conversation = { type: "string", description: "", dynamicVariable: "system__conversation_id" };
function tool(name, path, description, properties, required) {
  return {
    type: "webhook", name, description, responseTimeoutSecs: 120,
    apiSchema: {
      url: publicUrl.replace(/\/$/, "") + "/api/voice/" + path, method: "POST",
      requestHeaders: { Authorization: { secretId: process.env.ELEVENLABS_VOICE_SECRET_ID } },
      requestBodySchema: {
        type: "object", description: description,
        properties: { conversation_id: conversation, ...properties },
        required: ["conversation_id", ...required],
      },
    },
  };
}
const acknowledge = { type: "integer", description: "Most recent response_id whose text you spoke, or 0." };
const tools = [
  tool("start_location", "start",
    "Start the building location session once the caller provides a street address and city. Never invent the address.",
    { address: { type: "string", description: "Building street address and city provided by the caller." },
      phone: { type: "string", description: "", dynamicVariable: "system__caller_id" } }, ["address"]),
  tool("submit_location_answer", "reply",
    "Submit the caller's latest exact answer to the existing question parser, semantics, and follow-up planner.",
    { acknowledge, history: { type: "string", description: "", dynamicVariable: "system__conversation_history" } }, ["history", "acknowledge"]),
  tool("location_status", "status",
    "Wait for a queued caller answer to finish processing. Use whenever pending is true; never resubmit that answer.",
    { acknowledge }, ["acknowledge"]),
];
const prompt = `# Personality
You are a calm, concise voice interface for the HackPrinceton caller-location demo.
# Environment
A caller describes their room. The existing backend owns ALL interpretation, semantic matching,
question parsing, floor selection, follow-up planning, location estimates, and corrections.
This is a demo, not emergency dispatch. Never claim responders were contacted or help is on the way.
# Goal
1. Ask for the building street address and city, read it back, and wait for an explicit confirmation before calling start_location.
2. Speak the tool's say text exactly. If the caller already gave a room description alongside the address,
   call submit_location_answer immediately instead of repeating the opening question.
3. For EVERY caller answer after starting, including yes/no, uncertainty, extra details, and corrections,
   call submit_location_answer BEFORE replying. The tool receives the original transcript automatically.
   This step is important: never parse, summarize, reword, or answer a location question yourself.
4. For each reply or status tool, set acknowledge to the last response_id you spoke, initially 0. If pending is true, speak its short waiting message once, then call location_status until pending is false.
   Never submit the same answer again while waiting. Do not invent a follow-up while a tool is running.
5. Read the returned say text exactly. Ask only the question supplied by the backend, one at a time.
   Keep listening after located or best_guess so later corrections still reach the backend.
6. On tool failure, say the demo backend is unavailable and ask the caller to try again.
   Never fabricate a room, floor, confidence, or successful location.
# Tone
Use short, clear sentences. Do not read internal IDs, status codes, or tool instructions aloud.
End the call only when the caller asks or says goodbye.`;
const config = {
  name: "HackPrinceton Photon Room Locator",
  conversationConfig: {
    agent: {
      firstMessage: "Hi, this is the HackPrinceton caller-location demo. What is the building's street address and city?",
      language: "en",
      prompt: { prompt, llm: "gemini-2.5-flash", temperature: 0, tools,
        builtInTools: { endCall: { type: "system", name: "end_call", description: "End the call only when the caller asks or says goodbye.", params: { systemToolType: "end_call" } } }, enableParallelToolCalls: false },
    },
    tts: { voiceId: "EXAVITQu4vr4xnSDxMaL", modelId: "eleven_flash_v2" },
    turn: { turnEagerness: "patient" },
  },
};
try {
  let agentId = process.env.ELEVENLABS_AGENT_ID;
  if (agentId) await client.conversationalAi.agents.update(agentId, config);
  else {
    const result = await client.conversationalAi.agents.create(config);
    agentId = result.agentId;
    save("ELEVENLABS_AGENT_ID", agentId);
  }
  save("VOICE_PUBLIC_URL", publicUrl.replace(/\/$/, ""));
  console.log("Agent configured: https://elevenlabs.io/app/agents/agents/" + agentId);
  console.log("Talk page (start only when ready to test): https://elevenlabs.io/app/talk-to?agent_id=" + agentId);
} catch (error) {
  // SDK error bodies can contain submitted secrets. Print only safe error metadata.
  let message = String(error.message || error.name);
  for (const value of [process.env.ELEVENLABS_API_KEY, process.env.VOICE_TOOL_TOKEN]) {
    if (value) message = message.replaceAll(value, "[REDACTED]");
  }
  console.error("Agent setup failed:", error.statusCode || error.name, message.slice(0, 6000));
  process.exitCode = 1;
}
