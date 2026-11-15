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

async function optional(path) {
  const response = await fetch(path);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

export const getConfig = () => request("/api/config");
export const getAddresses = (q) => request(`/api/addresses?q=${encodeURIComponent(q)}`);
export const getDemoBuildings = () => request("/api/demo");
export const getSessions = () => request("/api/sessions");
export const getBuilding = (modelId) => request(`/api/buildings/${modelId}`);
// Map layers. Both are optional: a model without Matterport geolocation (or no Vexcel tile) answers 404, which
// resolves to null so the map view can say what's missing instead of erroring.
export const getBuildingGeo = (modelId) => optional(`/api/buildings/${modelId}/geo`);
export const getBuildingAerial = (modelId) => optional(`/api/buildings/${modelId}/aerial`);
export const getRoomShots =(sessionId) => request(`/api/sessions/${sessionId}/rooms`);
export const startCall = (phone, address, modelId = null) =>
  request("/api/calls", { method: "POST", body: JSON.stringify({ phone, address, model_id: modelId }) });
export const sendReply = (sessionId, text) =>
  request(`/api/sessions/${sessionId}/reply`, { method: "POST", body: JSON.stringify({ text }) });

// Live session snapshots. Reconnects automatically because the backend may restart during development. While the
// socket is down, the sessions are polled every 2 s instead, so the page never needs a reload to catch up.
export function subscribe(onSession, onStatus) {
  let ws;
  let closed = false;
  let live = false;
  const poll = setInterval(() => {
    if (live || closed) return;
    request("/api/sessions")
      .then((list) => list.forEach(onSession))
      .catch(() => {});
  }, 2000);
  const connect = () => {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${protocol}://${location.host}/ws`);
    ws.onopen = () => {
      live = true;
      onStatus(true);
    };
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "session") onSession(message.session);
    };
    ws.onclose = () => {
      live = false;
      onStatus(false);
      if (!closed) setTimeout(connect, 1500);
    };
  };
  connect();
  return () => {
    closed = true;
    clearInterval(poll);
    ws?.close();
  };
}
