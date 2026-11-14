import React, { useEffect, useMemo, useRef, useState } from "react";
import { getBuilding, getConfig, getRoomShots, getSessions, sendReply, startCall, subscribe } from "./api.js";
import FloorPlan from "./FloorPlan.jsx";

const DEMO_ADDRESS = "13122 Chipman Glen Dr, Houston, TX";
const STATUS_LABELS = {
  loading: "Loading building",
  listening: "Locating",
  located: "Located",
  best_guess: "Best guess",
  no_model: "No interior model",
  error: "Error",
};

export default function App() {
  const [config, setConfig] = useState(null);
  const [sessions, setSessions] = useState({});
  const [selectedId, setSelectedId] = useState(null);
  const [buildings, setBuildings] = useState({});
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getConfig().then(setConfig).catch((e) => setError(e.message));
    getSessions()
      .then((list) => setSessions(Object.fromEntries(list.map((s) => [s.id, s]))))
      .catch(() => {});
    return subscribe(
      (session) => setSessions((prev) => ({ ...prev, [session.id]: session })),
      setConnected,
    );
  }, []);

  const ordered = useMemo(() => Object.values(sessions).sort((a, b) => b.created - a.created), [sessions]);
  const selected = sessions[selectedId] || ordered[0] || null;

  // Fetch each building's geometry once, when a session first references it, and again when its orientation (north
  // and the nearest street, fetched in the background after the building loads) arrives.
  useEffect(() => {
    const modelId = selected?.model_id;
    const cached = buildings[modelId];
    const stale = cached && selected.orientation && !cached.orientation;
    if (!modelId || (cached && !stale) || selected.status === "loading") return;
    getBuilding(modelId)
      .then((b) => setBuildings((prev) => ({ ...prev, [modelId]: b })))
      .catch((e) => setError(e.message));
  }, [selected?.model_id, selected?.status, selected?.orientation, buildings]);

  async function handleCall(phone, address) {
    setError("");
    try {
      const session = await startCall(phone, address);
      setSessions((prev) => ({ ...prev, [session.id]: session }));
      setSelectedId(session.id);
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>Caller Locator</h1>
        <div className="topbar-meta">
          {config && (
            <span className="badge" title="How messages reach the caller">
              {config.messaging_mode === "photon" ? "iMessage via Photon" : "Simulated caller"}
            </span>
          )}
          {config && (
            <span className="badge" title="Who reads replies and words follow-up questions">
              {config.followups === "claude" ? "Follow-ups: Claude" : "Follow-ups: keyword rules"}
            </span>
          )}
          <span className={`conn ${connected ? "on" : "off"}`}>{connected ? "Live" : "Reconnecting…"}</span>
        </div>
      </header>

      <aside className="sidebar">
        <CallForm defaultPhone={config?.dev_test_phone || ""} onSubmit={handleCall} />
        {error && <p className="error">{error}</p>}
        <h2>Calls</h2>
        <ul className="session-list">
          {ordered.length === 0 && <li className="muted">No calls yet.</li>}
          {ordered.map((s) => (
            <li key={s.id}>
              <button className={s.id === selected?.id ? "session active" : "session"} onClick={() => setSelectedId(s.id)}>
                <span className="session-address">{s.address}</span>
                <span className="session-meta">
                  {s.phone} · <StatusPill status={s.status} />
                </span>
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main className="main">
        {selected ? (
          <SessionView session={selected} building={buildings[selected.model_id]} config={config} />
        ) : (
          <div className="empty">Start a simulated 911 call to begin.</div>
        )}
      </main>
    </div>
  );
}

function CallForm({ defaultPhone, onSubmit }) {
  const [phone, setPhone] = useState("");
  const [address, setAddress] = useState(DEMO_ADDRESS);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!phone && defaultPhone) setPhone(defaultPhone);
  }, [defaultPhone]);

  return (
    <form
      className="call-form"
      onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        await onSubmit(phone, address);
        setBusy(false);
      }}
    >
      <h2>Incoming 911 call</h2>
      <label>
        Caller phone
        <input value={phone} onChange={(e) => setPhone(e.target.value)} placeholder="+1 609 555 0100" required />
      </label>
      <label>
        Address
        <input value={address} onChange={(e) => setAddress(e.target.value)} required />
      </label>
      <button type="submit" className="primary" disabled={busy}>
        {busy ? "Starting…" : "Start call"}
      </button>
    </form>
  );
}

function StatusPill({ status }) {
  return <span className={`pill pill-${status}`}>{STATUS_LABELS[status] || status}</span>;
}

function SessionView({ session, building, config }) {
  const [showPhoto, setShowPhoto] = useState(false);
  // Room whose photos are shown; null follows the current best estimate.
  const [pickedRoom, setPickedRoom] = useState(null);
  // Floor shown on the plan; null follows the floor of the current best estimate.
  const [pickedFloor, setPickedFloor] = useState(null);
  useEffect(() => {
    setPickedRoom(null);
    setPickedFloor(null);
  }, [session.id]);
  const estimate = session.estimate;
  const shownRoom = pickedRoom ?? estimate?.room ?? null;
  const multiFloor = (building?.floors?.length || 0) > 1;
  const shownFloor = pickedFloor ?? estimate?.floor ?? 0;
  const floorplan = building?.floorplans?.[String(shownFloor)];

  function pickRoom(index) {
    setPickedRoom(index);
    const room = session.rooms.find((r) => r.index === index);
    if (room) setPickedFloor(room.floor); // show the floor the room is on
  }

  return (
    <div className="session-view">
      <section className="status-bar">
        <div>
          <StatusPill status={session.status} />
          <span className="status-detail">{session.status_detail}</span>
        </div>
        <div className="status-numbers">
          <span>
            Questions {session.questions_asked}/{session.max_questions}
          </span>
          {estimate && (
            <span>
              Top room <strong>{estimate.room_name}</strong>
              {multiFloor ? ` (${estimate.floor_label})` : ""} {pct(estimate.room_probability)} (next{" "}
              {pct(estimate.runner_up_probability)}) · spot ±{estimate.radius_m?.toFixed(1)} m
            </span>
          )}
        </div>
      </section>

      <section className="panel plan-panel">
        <div className="panel-head">
          <div>
            <h2>{building ? building.address : session.address}</h2>
            <MatchNote session={session} />
          </div>
          {floorplan && (
            <label className="toggle">
              <input type="checkbox" checked={showPhoto} onChange={(e) => setShowPhoto(e.target.checked)} />
              Aerial photo
            </label>
          )}
        </div>
        {building && multiFloor && (
          <FloorTabs
            floors={session.floors.length ? session.floors : building.floors}
            shown={shownFloor}
            estimateFloor={estimate?.floor}
            followsEstimate={pickedFloor === null}
            onPick={setPickedFloor}
          />
        )}
        {building ? (
          <FloorPlan
            building={building}
            session={session}
            floor={shownFloor}
            showPhoto={showPhoto}
            selectedRoom={shownRoom}
            onSelectRoom={pickRoom}
          />
        ) : (
          <div className="plan-placeholder">
            {session.status === "no_model" || session.status === "error"
              ? session.status_detail
              : "Loading interior model…"}
          </div>
        )}
        {building && (
          <RoomPhotos
            session={session}
            roomIndex={shownRoom}
            followsEstimate={pickedRoom === null}
            onFollowEstimate={() => setPickedRoom(null)}
          />
        )}
      </section>

      <section className="panel conversation-panel">
        <h2>Conversation</h2>
        <Conversation messages={session.messages} />
        <CallerBox session={session} mode={config?.messaging_mode} />
      </section>

      <section className="panel rooms-panel">
        <h2>Room probabilities</h2>
        {session.rooms.length === 0 ? (
          <p className="muted">Waiting for the building to load.</p>
        ) : (
          <ul className="room-bars">
            {session.rooms.map((room) => (
              <li key={room.index}>
                <button
                  type="button"
                  className={room.index === shownRoom ? "room-row selected" : "room-row"}
                  onClick={() => pickRoom(room.index)}
                  title="Show this room's photos"
                >
                  <span className="room-name">
                    {room.name}
                    {multiFloor && <span className="room-floor"> · {room.floor_label}</span>}
                  </span>
                  <span className="bar">
                    <span style={{ width: pct(room.probability) }} />
                  </span>
                  <span className="room-pct">{pct(room.probability)}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

// Every photo of one room, grouped by scan point, from GET /api/sessions/{id}/rooms. Photos don't change during a
// call, so they are fetched once per session/building; probabilities come from the live session snapshot.
function RoomPhotos({ session, roomIndex, followsEstimate, onFollowEstimate }) {
  const [rooms, setRooms] = useState(null);
  const [error, setError] = useState("");
  const ready = session.rooms.length > 0;
  useEffect(() => {
    if (!ready) return;
    setRooms(null);
    getRoomShots(session.id)
      .then((data) => setRooms(data.rooms))
      .catch((e) => setError(e.message));
  }, [session.id, session.model_id, ready]);

  if (error) return <p className="error">{error}</p>;
  if (!rooms || roomIndex === null) return null;
  const room = rooms.find((r) => r.index === roomIndex);
  if (!room) return null;
  const live = session.rooms.find((r) => r.index === roomIndex);

  return (
    <div className="views">
      <div className="views-head">
        <h3>
          {room.name} · {pct(live?.probability)}
          {room.area_m2 ? ` · ${room.area_m2} m²` : ""}
          {followsEstimate ? " (most likely room)" : ""}
        </h3>
        {!followsEstimate && (
          <button type="button" className="link" onClick={onFollowEstimate}>
            Back to most likely room
          </button>
        )}
      </div>
      {room.sweeps.length === 0 ? (
        <p className="muted">No photos were taken in this room. It can only be identified from what the caller says.</p>
      ) : (
        room.sweeps.map((sweep, i) => (
          <div key={sweep.candidate} className="sweep-shots">
            <span className="muted">
              {i === 0 && followsEstimate && room.sweeps.length > 1
                ? `Most likely spot (${pct(liveScan(session, sweep.candidate))})`
                : `Scan point ${i + 1} of ${room.sweeps.length}`}
            </span>
            <div className="view-strip">
              {sweep.views.map((url) => (
                <a key={url} href={url} target="_blank" rel="noreferrer">
                  <img src={url} alt={`${room.name}, scan point ${i + 1}`} loading="lazy" />
                </a>
              ))}
            </div>
          </div>
        ))
      )}
    </div>
  );
}

// One tab per floor with the chance the caller is on it. The plan follows the most likely floor until the
// dispatcher picks one.
function FloorTabs({ floors, shown, estimateFloor, followsEstimate, onPick }) {
  return (
    <div className="floor-tabs" role="tablist" aria-label="Floors">
      {[...floors].reverse().map((f) => (
        <button
          key={f.index}
          type="button"
          role="tab"
          aria-selected={f.index === shown}
          className={f.index === shown ? "floor-tab active" : "floor-tab"}
          onClick={() => onPick(f.index === estimateFloor && !followsEstimate ? null : f.index)}
          title={f.index === estimateFloor ? "Most likely floor" : ""}
        >
          {f.label}
          {f.probability != null && <span className="floor-pct"> {pct(f.probability)}</span>}
        </button>
      ))}
    </div>
  );
}

// Live probability of one scan point (the photos list is fetched once; probabilities change every reply).
const liveScan = (session, candidate) => session.candidate_probabilities?.[candidate] ?? 0;

// Which Matterport property the call is using. A non-exact match is highlighted so the dispatcher confirms it.
function MatchNote({ session }) {
  const match = session.match;
  if (!match) return null;
  if (match.exact) return <p className="match match-exact">Exact address match</p>;
  return (
    <p className="match match-approx" title={`Caller address: ${session.address}`}>
      Closest match to “{session.address}” · {pct(match.score)} similar. Confirm with the caller.
    </p>
  );
}

function Conversation({ messages }) {
  const end = useRef(null);
  useEffect(() => end.current?.scrollIntoView({ block: "end" }), [messages.length]);
  return (
    <ol className="messages">
      {messages.map((m, i) => (
        <li key={i} className={`msg msg-${m.role}`}>
          <span className="msg-who">{m.role === "dispatch" ? "To caller" : m.role === "caller" ? "Caller" : "System"}</span>
          <span className="msg-text">{m.text}</span>
        </li>
      ))}
      <li ref={end} />
    </ol>
  );
}

function CallerBox({ session, mode }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const open = session.status === "listening" || session.status === "loading";
  const quick = ["Yes", "No", "I don't know"];

  async function send(value) {
    if (!value.trim()) return;
    setBusy(true);
    try {
      await sendReply(session.id, value);
      setText("");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      className="caller-box"
      onSubmit={(e) => {
        e.preventDefault();
        send(text);
      }}
    >
      <label htmlFor="caller-input">
        Simulated caller reply{mode === "photon" ? " (overrides iMessage for testing)" : ""}
      </label>
      <div className="caller-row">
        <input
          id="caller-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={open ? "e.g. There's a stove and a fridge" : "Call is no longer collecting replies"}
          disabled={!open || busy}
        />
        <button type="submit" className="primary" disabled={!open || busy || !text.trim()}>
          Send
        </button>
      </div>
      <div className="quick">
        {quick.map((q) => (
          <button type="button" key={q} disabled={!open || busy} onClick={() => send(q)}>
            {q}
          </button>
        ))}
      </div>
    </form>
  );
}

const pct = (p) => `${Math.round((p || 0) * 100)}%`;
