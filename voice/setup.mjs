// Provision the real ElevenLabs agent. This command never starts a conversation or places a call.
import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { randomBytes } from "node:crypto";

// Rewrite NAME=... in place (or add it once). Appending left duplicates, and a blank `VOICE_TOOL_TOKEN=` above the
// real line shadowed it in readers that keep the first value. Later duplicates are dropped; comments and other
// lines are kept as they are.
function setEnvLine(text, name, value) {
  const pattern = new RegExp("^\\s*(export\\s+)?" + name + "\\s*=");
  let found = false;
  const lines = text.split(/\r?\n/).filter(line => {
    if (!pattern.test(line)) return true;
    if (found) return false;
    found = true;
    return true;
  }).map(line => pattern.test(line) ? name + "=" + value : line);
  if (!found) {
    while (lines.length && lines[lines.length - 1] === "") lines.pop();
    lines.push(name + "=" + value);
  }
  return lines.join("\n").replace(/\n*$/, "\n");
}

function save(name, value) {
  const file = new URL("../.env", import.meta.url);
  writeFileSync(file, setEnvLine(existsSync(file) ? readFileSync(file, "utf8") : "", name, value));
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
    type: "new", name: "LifeLine " + randomBytes(4).toString("hex"),
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
    "Start the location session as soon as the caller gives a building address. Call it once per call. Never invent the address.",
    { address: { type: "string", description: "The building address as the caller said it, with the house number in digits (\"13122\", not \"thirteen one twenty-two\"), plus the street, and the city if given." },
      phone: { type: "string", description: "", dynamicVariable: "system__caller_id" } }, ["address"]),
  tool("submit_location_answer", "reply",
    "Submit the caller's latest exact answer, including answers about the address (a repeated address, yes or no to \"Did you mean...\", or which of two addresses).",
    { acknowledge, history: { type: "string", description: "", dynamicVariable: "system__conversation_history" } }, ["history", "acknowledge"]),
  tool("location_status", "status",
    "Wait for a queued caller answer to finish processing. Use whenever pending is true; never resubmit that answer.",
    { acknowledge }, ["acknowledge"]),
];
const prompt = `# Personality
You are LifeLine, a calm voice interface that helps locate a caller inside a building.
# Environment
A caller describes their room. The backend owns ALL interpretation: the address lookup, question parsing,
floor selection, follow-up planning, location estimates, and corrections.
Use the backend's wording and do not invent claims about dispatch or help arriving.
# Goal
1. Get the building address. As soon as the caller gives one, call start_location once with it. Write the house
   number in digits ("13122 Chipman Glen Drive, Houston"), never in words. Do not read the address back first:
   the backend checks it and asks the caller if it is unsure.
2. Speak the tool's say text exactly. If the caller already gave a room description alongside the address,
   call submit_location_answer immediately instead of repeating the opening question.
3. If the backend asks for the address again, or asks "Did you mean ...?" or "Is it ... or ...?", relay the
   caller's answer with submit_location_answer. Never call start_location a second time.
4. For EVERY caller answer after starting, including yes/no, uncertainty, extra details, and corrections,
   call submit_location_answer BEFORE replying. The tool receives the original transcript automatically.
   This step is important: never parse, summarize, reword, or answer a location question yourself.
5. For each reply or status tool, set acknowledge to the last response_id you spoke, initially 0. If pending is true, speak its short waiting message once, then call location_status until pending is false.
   Never submit the same answer again while waiting. Do not invent a follow-up while a tool is running.
6. Read the returned say text exactly. Ask only the question supplied by the backend, one at a time.
   Keep listening after located or best_guess so later corrections still reach the backend.
7. When a tool response has ended set to true, say its say text exactly, then call end_call. Say nothing after it.
8. On tool failure, say the location service is unavailable and ask the caller to try again.
   Never fabricate a room, floor, confidence, or successful location.
# Tone
Use short, clear sentences. Do not read internal IDs, status codes, or tool instructions aloud.
End the call when a tool response has ended set to true, or when the caller asks or says goodbye.`;
const config = {
  name: "LifeLine",
  conversationConfig: {
    agent: {
      firstMessage: "This is LifeLine. What is the building’s street address and city?",
      language: "en",
      prompt: { prompt, llm: "gemini-2.5-flash", temperature: 0, tools,
        builtInTools: { endCall: { type: "system", name: "end_call", description: "End the call after speaking a tool response whose ended is true, or when the caller asks or says goodbye.", params: { systemToolType: "end_call" } } }, enableParallelToolCalls: false },
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
