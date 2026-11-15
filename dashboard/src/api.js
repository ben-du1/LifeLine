async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

export const getConfig = () => request("/api/config");
export const getSessions = () => request("/api/sessions");
export const getBuilding = (modelId) => request(`/api/buildings/${modelId}`);
export const getRoomShots = (sessionId) => request(`/api/sessions/${sessionId}/rooms`);
export const startCall = (phone, address) =>
  request("/api/calls", { method: "POST", body: JSON.stringify({ phone, address }) });
// Dispatcher: send `text` as their own question, or (no text) let the system ask the next best question.
export const askQuestion = (sessionId, text = "") =>
  request(`/api/sessions/${sessionId}/ask`, { method: "POST", body: JSON.stringify({ text }) });
export const sendReply = (sessionId, text) =>
  request(`/api/sessions/${sessionId}/reply`, { method: "POST", body: JSON.stringify({ text }) });

// Live session snapshots. Reconnects automatically because the backend may restart during development.
export function subscribe(onSession, onStatus) {
  let ws;
  let closed = false;
  const connect = () => {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${protocol}://${location.host}/ws`);
    ws.onopen = () => onStatus(true);
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "session") onSession(message.session);
    };
    ws.onclose = () => {
      onStatus(false);
      if (!closed) setTimeout(connect, 1500);
    };
  };
  connect();
  return () => {
    closed = true;
    ws?.close();
  };
}
