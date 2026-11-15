import React, { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { getBuilding, getConfig, getRoomShots, getSessions, sendInbound, sendReply, startCall, subscribe } from "./api.js";
import { getAddresses } from "./api.js";
import FloorPlan from "./FloorPlan.jsx";
import ApproachCard from "./ApproachCard.jsx";
import { checkOrder, confidenceWord, describeEffect, pct } from "./confidence.js";

// Leaflet is only needed once someone opens the map, so it is loaded then.
const MapView = lazy(() => import("./MapView.jsx"));
const AddressMap = lazy(() => import("./AddressMap.jsx"));

// The screen is for the search and rescue team: nobody types into it during a real call. The tools that start or
// answer calls by hand are kept for testing behind ?test=1.
const TEST_MODE = new URLSearchParams(window.location.search).get("test") === "1";
const SAMPLE_ADDRESS = "13122 Chipman Glen Dr, Houston, TX";
// A new call takes the screen only if nobody picked a call in this long, so it never yanks away a call being read.
const AUTO_SELECT_IDLE_MS = 60_000;
const NEW_CALL_ALERT_MS = 8_000;
const STATUS_LABELS = {
  awaiting_address: "Getting address",
  loading: "Loading building",
  listening: "Locating",
  located: "Located",
  best_guess: "Best guess",
  out: "Out of building",
  transferred: "Transferred to 911",
  no_model: "No interior model",
  error: "Error",
};

function mergeSession(sessions, incoming) {
  const current = sessions[incoming.id];
  if (current && (current.revision ?? 0) > (incoming.revision ?? 0)) return sessions;
  return { ...sessions, [incoming.id]: incoming };
}

// Calls that change what the team does right now: the caller got out, is in danger, or was handed off.
function needsAttention(s) {
  return s.status === "out" || s.status === "transferred" || Boolean(s.danger);
}

// "+1 678 … 08": enough to tell calls apart on a shared screen without showing the whole number.
function maskPhone(phone) {
  if (!phone) return "Unknown caller";
  const digits = phone.replace(/\D/g, "");
  if (digits.length < 7) return phone;
  const country = phone.startsWith("+1") ? "+1 " : phone.startsWith("+") ? "+" : "";
  const local = country === "+1 " ? digits.slice(1) : digits;
  return `${country}${local.slice(0, 3)} … ${local.slice(-2)}`;
}

function minutesSince(created, now) {
  const minutes = Math.max(0, Math.floor((now - created * 1000) / 60_000));
  return minutes < 1 ? "just now" : `${minutes} min`;
}

// Re-render every 30 s so "minutes since start" stays current without any new snapshot.
function useNow(intervalMs = 30_000) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs]);
  return now;
}

class SessionBoundary extends React.Component {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (this.state.failed) {
      return (
        <section className="panel" role="alert">
          <p>Unable to display this call.</p>
          <button onClick={() => this.setState({ failed: false })}>Retry call view</button>
        </section>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  const [config, setConfig] = useState(null);
  const [sessions, setSessions] = useState({});
  const [selectedId, setSelectedId] = useState(null);
  const [buildings, setBuildings] = useState({});
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState("");
  const [alert, setAlert] = useState(null); // {id, text} for the short "New call" banner
  const now = useNow();
  const lastPick = useRef(0);
  const known = useRef(new Set());
  // Calls that already existed when the page opened are not "new"; only ones created after it are announced.
  const openedAt = useRef(Date.now());

  useEffect(() => {
    getConfig().then(setConfig).catch((e) => setError(e.message));
    getSessions()
      .then((list) => setSessions((prev) => list.reduce(mergeSession, prev)))
      .catch(() => {});
    return subscribe(
      (session) => setSessions((prev) => mergeSession(prev, session)),
      setConnected,
      () => { setSessions({}); setSelectedId(null); setBuildings({}); known.current = new Set(); },
    );
  }, []);

  const ordered = useMemo(
    () => Object.values(sessions).sort((a, b) => (needsAttention(b) - needsAttention(a)) || b.created - a.created),
    [sessions],
  );
  const selected = sessions[selectedId] || ordered[0] || null;
  const active = ordered.filter((s) => !s.ended).length;

  function pick(id) {
    lastPick.current = Date.now();
    setSelectedId(id);
  }

  // A call that arrives by itself: show it if nobody is reading another call, and announce it either way.
  useEffect(() => {
    for (const s of Object.values(sessions)) {
      if (known.current.has(s.id)) continue;
      known.current.add(s.id);
      if (s.created * 1000 < openedAt.current - 5_000) continue;
      const idle = Date.now() - lastPick.current > AUTO_SELECT_IDLE_MS;
      if (idle) setSelectedId(s.id);
      const who = s.channel === "voice" ? "Voice call" : "Text";
      setAlert({ id: s.id, text: `New call: ${who} from ${maskPhone(s.phone)}`, shown: idle });
    }
  }, [sessions]);

  useEffect(() => {
    if (!alert) return undefined;
    const timer = setTimeout(() => setAlert(null), NEW_CALL_ALERT_MS);
    return () => clearTimeout(timer);
  }, [alert]);

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

  return (
    <div className="app">
      <header className="topbar">
        <h1 className="brand">
          <span className="brand-mark" aria-hidden="true"><span /></span>
          LifeLine
        </h1>
        <div className="topbar-meta">
          <span className="active-count">{active === 1 ? "1 active call" : `${active} active calls`}</span>
          <span className={`conn ${connected ? "on" : "off"}`}>{connected ? "Live" : "Reconnecting"}</span>
        </div>
      </header>

      <aside className="sidebar">
        <h2 className="side-title calls-title">Calls</h2>
        {error && <p className="error">{error}</p>}
        <ul className="session-list">
          {ordered.length === 0 && <li className="muted">Waiting for calls. A call appears here as soon as someone texts or calls the line.</li>}
          {ordered.map((s) => (
            <li key={s.id}>
              <CallRow session={s} active={s.id === selected?.id} now={now} onPick={() => pick(s.id)} />
            </li>
          ))}
        </ul>
        {TEST_MODE && (
          <TestTools
            config={config}
            selected={selected}
            onSession={(id, snapshot) => {
              if (snapshot) setSessions((prev) => mergeSession(prev, snapshot));
              if (id) pick(id);
            }}
          />
        )}
      </aside>

      <main className="main">
        {alert && (
          <div className="new-call-alert" role="status">
            <span>{alert.text}</span>
            {selected?.id !== alert.id && (
              <button type="button" onClick={() => { pick(alert.id); setAlert(null); }}>Show</button>
            )}
          </div>
        )}
        {selected ? (
          <SessionBoundary key={selected.id}>
            <SessionView session={selected} building={buildings[selected.model_id]} />
          </SessionBoundary>
        ) : (
          <div className="empty">
            <div className="empty-copy">
              <h2>No active call</h2>
              <p>Calls and texts to the line appear here by themselves.</p>
            </div>
            <div className="empty-grid"><span className="map-crosshair" /></div>
          </div>
        )}
      </main>
    </div>
  );
}

function CallRow({ session: s, active, now, onPick }) {
  const flags = [];
  if (s.danger) flags.push("Danger");
  const classes = ["session", active && "active", needsAttention(s) && "attention", s.ended && "ended"].filter(Boolean);
  return (
    <button className={classes.join(" ")} onClick={onPick}>
      <span className="session-top">
        <span className={`badge badge-${s.channel === "voice" ? "voice" : "text"}`}>{s.channel === "voice" ? "Voice" : "Text"}</span>
        <span className="session-phone">{maskPhone(s.phone)}</span>
        <span className="session-age">{minutesSince(s.created, now)}</span>
      </span>
      <span className={s.address ? "session-address" : "session-address pending"}>{s.address || "Address pending"}</span>
      <span className="session-meta">
        <StatusPill status={s.status} />
        {flags.map((f) => <span key={f} className="flag">{f}</span>)}
        {s.ended && <span>Ended</span>}
      </span>
    </button>
  );
}

// Hidden test tools (?test=1): start a call the way the line would, or answer as the caller.
function TestTools({ config, selected, onSession }) {
  return (
    <details className="test-tools" open>
      <summary>Test tools</summary>
      <InboundForm defaultPhone={config?.dev_test_phone || ""} onSession={onSession} />
      <CallForm defaultPhone={config?.dev_test_phone || ""} onSession={onSession} />
      {selected && <CallerBox session={selected} mode={config?.messaging_mode} />}
    </details>
  );
}

function InboundForm({ defaultPhone, onSession }) {
  const [phone, setPhone] = useState("");
  const [text, setText] = useState("Help, there's smoke in the house");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    if (defaultPhone) setPhone((value) => value || defaultPhone);
  }, [defaultPhone]);
  return (
    <form
      className="call-form"
      onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        setError("");
        try {
          const result = await sendInbound(phone, text);
          onSession(result.session, null);
        } catch (err) {
          setError(err.message); // e.g. "No session for this phone number" until unknown numbers open calls
        } finally {
          setBusy(false);
        }
      }}
    >
      <h3 className="side-title">Simulate incoming text</h3>
      <label>
        From
        <input value={phone} onChange={(e) => setPhone(e.target.value)} placeholder="+1 555 010 0100" required />
      </label>
      <label>
        Message
        <input value={text} onChange={(e) => setText(e.target.value)} required />
      </label>
      <button type="submit" className="primary" disabled={busy}>{busy ? "Sending…" : "Send text"}</button>
      {error && <p className="error">{error}</p>}
    </form>
  );
}

function CallForm({ defaultPhone, onSession }) {
  const [phone, setPhone] = useState("");
  const [address, setAddress] = useState(SAMPLE_ADDRESS);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const sameAddress = (a, b) => a.trim().toLowerCase() === b.trim().toLowerCase();
  useEffect(() => {
    if (defaultPhone) setPhone((value) => value || defaultPhone);
  }, [defaultPhone]);
  // Suggestions from all ~10,000 callable addresses in the hackathon's model index (GET /api/addresses).
  const [suggestions, setSuggestions] = useState([]);
  useEffect(() => {
    const q = address.trim();
    if (q.length < 3) {
      setSuggestions([]);
      return undefined;
    }
    let current = true;
    const timer = setTimeout(() => {
      getAddresses(q).then((results) => {
        if (current) setSuggestions(results);
      }).catch(() => {
        if (current) setSuggestions([]);
      });
    }, 150);
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [address]);
  const picked = suggestions.find((s) => sameAddress(s.address, address));

  return (
    <form
      className="call-form"
      onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        setError("");
        try {
          const session = await startCall(phone, address, picked?.model_id || null);
          onSession(session.id, session);
        } catch (err) {
          setError(err.message);
        } finally {
          setBusy(false);
        }
      }}
    >
      <h3 className="side-title">Simulate a call with address</h3>
      <label>
        Caller phone
        <input value={phone} onChange={(e) => setPhone(e.target.value)} placeholder="+1 555 010 0100" required />
      </label>
      <label>
        Address
        <input
          value={address}
          onChange={(e) => setAddress(e.target.value)}
          list="address-suggestions"
          autoComplete="off"
          placeholder="Any Houston or Los Angeles address"
          required
        />
        <datalist id="address-suggestions">
          {suggestions.map((s) => (
            <option key={s.model_id} value={s.address}>
              {`${s.segment === "residential" ? "House" : "Commercial"}, ${s.rooms ?? "?"} rooms${s.floors > 1 ? `, ${s.floors} floors` : ""}${s.cached ? ", ready" : ""}`}
            </option>
          ))}
        </datalist>
      </label>
      {(suggestions.length > 0 || picked) && (
        <Suspense fallback={<div className="address-map" />}>
          <AddressMap suggestions={suggestions} picked={picked} onPick={(s) => setAddress(s.address)} />
        </Suspense>
      )}
      <button type="submit" className="primary" disabled={busy}>
        {busy ? "Starting…" : "Start call"}
      </button>
      {error && <p className="error">{error}</p>}
    </form>
  );
}

function capitalize(text) {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}

function StatusPill({ status }) {
  return <span className={`pill pill-${status}`}>{STATUS_LABELS[status] || status}</span>;
}

// The approach card's side and street in a few words ("rear (north) side, away from Pointer St"): the first sentence
// of its text without the floor, which the headline already shows. Hidden until the room is reasonably settled, the
// same rule ApproachCard uses.
function approachLine(session) {
  const card = session.approach;
  const p = session.estimate?.room_probability ?? 0;
  if (!card?.text || !(session.status === "located" || session.status === "best_guess" || p >= 0.5)) return null;
  const first = card.text.split(". ")[0];
  const prefix = `${card.floor_text}, `;
  if (!first.startsWith(prefix)) return null; // no compass direction known: nothing beyond the floor
  return first.slice(prefix.length);
}

// Rooms to search in order, when one room is not enough: a best guess, or rooms the backend can't tell apart.
function searchOrder(session) {
  const tie = session.tie_group || [];
  if (tie.length >= 2) return tie.map((r) => r.name || r.room_name).filter(Boolean).join(", then ") || null;
  if (session.status === "best_guess") return checkOrder(session.rooms);
  return null;
}

function latestCallerText(session) {
  const last = [...(session.messages || [])].reverse().find((m) => m.role === "caller");
  return last?.text || null;
}

// The top of the call: where to go, in signage type. Each status that has no room yet says what is happening instead.
function Headline({ session }) {
  const estimate = session.estimate;
  const multiFloor = (session.floors?.length || 0) > 1;
  const danger = session.danger && <p className="headline-danger" role="alert">Caller reported danger</p>;
  const ended = session.ended && <span className="headline-ended">Call ended</span>;
  const caller = (
    <p className="headline-caller">
      <span className={`badge badge-${session.channel === "voice" ? "voice" : "text"}`}>{session.channel === "voice" ? "Voice" : "Text"}</span>
      {session.phone || "Unknown caller"}
      {ended}
    </p>
  );

  if (session.status === "transferred") {
    const transfer = session.transfer || {};
    return (
      <section className="incident incident-transferred">
        <div className="incident-where">
          {caller}
          <p className="incident-room">Outside service area: handed off to standard 911</p>
          {transfer.reason && <p className="incident-zone">{capitalize(transfer.reason)}</p>}
          {transfer.stub !== false && (
            <p className="stub-note">
              Hand-off recorded{transfer.target ? ` to ${transfer.target}` : ""}
              {transfer.ts ? ` at ${new Date(transfer.ts * 1000).toLocaleTimeString()}` : ""}. No live line is connected in
              this build.
            </p>
          )}
          {danger}
        </div>
      </section>
    );
  }

  if (session.status === "out") {
    const text = latestCallerText(session);
    return (
      <section className="incident incident-out">
        <div className="incident-where">
          {caller}
          <p className="incident-room">Caller is out of the building</p>
          {text && <p className="incident-zone">Latest text: “{text}”</p>}
          {danger}
        </div>
      </section>
    );
  }

  if (!estimate) {
    const words = {
      awaiting_address: "Getting the address from the caller",
      loading: "Loading the building",
      listening: "Locating the caller",
      no_model: "No interior model for this address",
      error: "Something went wrong with this call",
    };
    return (
      <section className={`incident incident-${session.status}`}>
        <div className="incident-where">
          {caller}
          <p className="incident-room incident-waiting">{words[session.status] || STATUS_LABELS[session.status] || session.status}</p>
          {session.status_detail && <p className="incident-zone">{session.status_detail}</p>}
          {danger}
        </div>
      </section>
    );
  }

  const { word, tone } = confidenceWord(session.status, estimate.room_probability);
  const approach = approachLine(session);
  const order = searchOrder(session);
  return (
    <section className={`incident incident-${session.status}`}>
      <div className="incident-where">
        {caller}
        <p className="incident-room">
          <span className="go-to">Go to</span>
          {multiFloor && <span className="go-floor">{estimate.floor_label} · </span>}
          {estimate.room_name}
        </p>
        {approach && <p className="incident-approach">{capitalize(approach)}</p>}
        {estimate.zone_label && <p className="incident-zone">{capitalize(estimate.zone_label)}</p>}
        {order && (
          <p className="check-first">
            Search order: <strong>{order}</strong>
          </p>
        )}
        {danger}
      </div>
      <div className="incident-numbers">
        <p className="incident-odds">
          <span className="incident-pct">{pct(estimate.room_probability)}</span>
          <span className={`conf conf-${tone}`}>{word}</span>
          <span className="muted">next best {pct(estimate.runner_up_probability)}</span>
        </p>
      </div>
    </section>
  );
}

function SessionView({ session, building }) {
  const [showPhoto, setShowPhoto] = useState(false);
  const [planMode, setPlanMode] = useState("plan"); // "plan" (schematic) or "map" (real surroundings)
  // Room whose photos are shown; null follows the current best estimate.
  const [pickedRoom, setPickedRoom] = useState(null);
  // Floor shown on the plan; null follows the floor of the current best estimate.
  const [pickedFloor, setPickedFloor] = useState(null);
  useEffect(() => {
    setPickedRoom(null);
    setPickedFloor(null);
  }, [session.id]);
  const estimate = session.estimate;
  const rooms = session.rooms || [];
  const shownRoom = pickedRoom ?? estimate?.room ?? null;
  const multiFloor = (building?.floors?.length || 0) > 1;
  const shownFloor = pickedFloor ?? estimate?.floor ?? 0;
  const floorplan = building?.floorplans?.[String(shownFloor)];
  const noBuildingYet = session.status === "awaiting_address" || session.status === "transferred";

  function pickRoom(index) {
    setPickedRoom(index);
    const room = rooms.find((r) => r.index === index);
    if (room) setPickedFloor(room.floor); // show the floor the room is on
  }

  return (
    <div className="session-view">
      <Headline session={session} />

      <section className="panel plan-panel">
        <div className="panel-head">
          <div>
            <h2>{building ? building.address : session.address || "Address pending"}</h2>
            <MatchNote session={session} />
          </div>
        </div>
        <div className="plan-controls">
          <div className="panel-tools">
            {floorplan && planMode === "plan" && (
              <label className="toggle">
                <input type="checkbox" checked={showPhoto} onChange={(e) => setShowPhoto(e.target.checked)} />
                Aerial photo
              </label>
            )}
            {building && (
              <div className="segmented" role="group" aria-label="Plan view">
                <button type="button" aria-pressed={planMode === "plan"} className={planMode === "plan" ? "active" : ""} onClick={() => setPlanMode("plan")}>
                  Plan
                </button>
                <button type="button" aria-pressed={planMode === "map"} className={planMode === "map" ? "active" : ""} onClick={() => setPlanMode("map")}>
                  Map
                </button>
              </div>
            )}
          </div>
          {building && multiFloor && (
            <FloorTabs
              floors={session.floors?.length ? session.floors : building.floors}
              shown={shownFloor}
              estimateFloor={estimate?.floor}
              followsEstimate={pickedFloor === null}
              onPick={setPickedFloor}
            />
          )}
        </div>
        {building && planMode === "map" ? (
          <Suspense fallback={<div className="plan-placeholder">Loading map…</div>}>
            <MapView
              building={building}
              session={session}
              floor={shownFloor}
              selectedRoom={shownRoom}
              onSelectRoom={pickRoom}
            />
          </Suspense>
        ) : building ? (
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
            {session.status === "awaiting_address"
              ? "The floor plan appears once the caller's address is matched."
              : session.status === "transferred"
                ? "No interior model: this address is outside the service area."
                : session.status === "no_model" || session.status === "error"
                  ? session.status_detail
                  : "Loading interior model…"}
          </div>
        )}
        {building && <ApproachCard session={session} />}
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
        <h2>Caller said</h2>
        <Conversation messages={session.messages || []} evidence={session.evidence || []} />
      </section>

      <section className="panel rooms-panel">
        <h2>Room probabilities</h2>
        {rooms.length === 0 ? (
          <p className="muted">{noBuildingYet ? "No building yet." : "Waiting for the building to load."}</p>
        ) : !estimate && session.status === "listening" ? (
          <p className="muted">No room evidence yet.</p>
        ) : (
          <RoomBars session={session} shownRoom={shownRoom} multiFloor={multiFloor} onPick={pickRoom} />
        )}
      </section>
    </div>
  );
}

// The room list, most likely first. Rooms the backend can't tell apart yet (snapshot `tie_group`, when present) are
// bracketed together so nobody reads the order inside the group as meaningful.
function RoomBars({ session, shownRoom, multiFloor, onPick }) {
  const tie = (session.tie_group || []).length >= 2 ? new Set(session.tie_group.map((r) => r.index)) : new Set();
  const topIndex = session.rooms[0]?.index;
  const row = (room) => (
    <RoomRow
      key={room.index}
      room={room}
      selected={room.index === shownRoom}
      multiFloor={multiFloor}
      onPick={onPick}
      confidence={room.index === topIndex && !tie.has(room.index) ? { status: session.status } : null}
    />
  );
  const items = [];
  let tieShown = false;
  for (const room of session.rooms) {
    if (!tie.has(room.index)) {
      items.push(row(room));
    } else if (!tieShown) {
      tieShown = true;
      items.push(
        <li key="tie" className="tie-group">
          <span className="tie-label">Can't tell apart yet</span>
          <ul className="room-bars">{session.rooms.filter((r) => tie.has(r.index)).map(row)}</ul>
        </li>,
      );
    }
  }
  return <ul className="room-bars">{items}</ul>;
}

function RoomRow({ room, selected, multiFloor, onPick, confidence }) {
  return (
    <li>
      <button
        type="button"
        className={selected ? "room-row selected" : "room-row"}
        onClick={() => onPick(room.index)}
        title="Show this room's photos"
      >
        <span className="room-name">
          <span className="room-name-text">
            {room.name}
            {multiFloor && <span className="room-floor">{room.floor_label}</span>}
          </span>
          {room.has_photos === false && (
            <span className="room-badge muted-badge" title="Found from the caller's words only">
              no photos
            </span>
          )}
          {confidence && <ConfidenceTag status={confidence.status} probability={room.probability} />}
        </span>
        <span className="bar">
          <span style={{ width: `${Math.round((room.probability || 0) * 100)}%` }} />
        </span>
        <span className="room-pct">{pct(room.probability)}</span>
      </button>
    </li>
  );
}

function ConfidenceTag({ status, probability }) {
  const { word, tone } = confidenceWord(status, probability);
  return <span className={`conf conf-${tone}`}>{word}</span>;
}

// Every photo of one room, grouped by scan point, from GET /api/sessions/{id}/rooms. Re-fetched after each message,
// since the scan points are ordered by the live estimate; the panel keeps the old photos until the new list arrives.
function RoomPhotos({ session, roomIndex, followsEstimate, onFollowEstimate }) {
  const [rooms, setRooms] = useState(null);
  const [error, setError] = useState("");
  const ready = session.rooms.length > 0;
  useEffect(() => {
    setRooms(null);
  }, [session.id]);
  useEffect(() => {
    if (!ready) return;
    setError("");
    let current = true;
    getRoomShots(session.id)
      .then((data) => { if (current) setRooms(data.rooms); })
      .catch((e) => { if (current) setError(e.message); });
    return () => { current = false; };
  }, [session.id, session.model_id, ready, session.messages.length]);

  if (error) return <p className="error">{error}</p>;
  if (!rooms || roomIndex === null) return null;
  const room = rooms.find((r) => r.index === roomIndex);
  if (!room) return null;
  const live = session.rooms.find((r) => r.index === roomIndex);

  return (
    <div className="views">
      <div className="views-head">
        <h3>
          Photos of {room.name}
          <span className="views-sub">
            {followsEstimate ? "most likely room, " : ""}
            {pct(live?.probability)}
            {room.area_m2 ? `, ${room.area_m2} m²` : ""}
          </span>
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
              Scan point {i + 1} of {room.sweeps.length}
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
// viewer picks one.
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

// Which Matterport property the call is using. A non-exact match is highlighted so it gets confirmed.
function MatchNote({ session }) {
  const match = session.match;
  if (!match || match.exact || !session.address) return null;
  return (
    <p className="match match-approx" title={`Caller address: ${session.address}`}>
      Closest match to “{session.address}” ({pct(match.score)} similar). Check the address before relying on it.
    </p>
  );
}

// Each caller message is followed by its effect on the room probabilities, from the snapshot's evidence log. Entries
// are matched by timestamp, then by text (each entry used once) in case a backend rounds the timestamp.
function Conversation({ messages, evidence = [] }) {
  const list = useRef(null);
  useEffect(() => {
    if (list.current) list.current.scrollTop = list.current.scrollHeight;
  }, [messages.length, evidence.length]);
  const used = new Set();
  const effectFor = (m) => {
    let entry = evidence.find((e, i) => !used.has(i) && (m.id ? e.message_id === m.id : e.ts === m.ts && e.text === m.text));
    entry ??= evidence.find((e, i) => !used.has(i) && e.text === m.text);
    if (entry) used.add(evidence.indexOf(entry));
    return entry;
  };
  return (
    <ol className="messages" ref={list}>
      {messages.map((m, i) => {
        const entry = m.role === "caller" ? effectFor(m) : null;
        return (
          <li key={i} className={`msg msg-${m.role}`}>
            <span className="msg-who">{m.role === "dispatch" ? "To caller" : m.role === "caller" ? "Caller" : "Note"}</span>
            <span className="msg-text">{m.text}</span>
            {entry && (
              <span className="msg-effect" title={`Top room before this message: ${entry.top_before}`}>
                {entry.note || describeEffect(entry)}
              </span>
            )}
          </li>
        );
      })}
    </ol>
  );
}

function CallerBox({ session, mode }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const open = !session.ended && ["awaiting_address", "listening", "loading", "located", "best_guess"].includes(session.status) && session.channel !== "voice";
  const quick = ["Yes", "No", "I don't know"];

  async function send(value) {
    if (!value.trim()) return;
    setText("");   // the message shows in the conversation right away (live update); no need to wait for the reply
    setBusy(true);
    try {
      await sendReply(session.id, value);
    } catch {
      setText(value);   // give the text back if it didn't go through
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
      <h3 className="side-title">Reply as the caller</h3>
      <label htmlFor="caller-input">
        To {session.phone || "unknown caller"}{mode === "photon" ? " (bypasses iMessage)" : ""}
      </label>
      <div className="caller-row">
        <input
          id="caller-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={open ? "e.g. There's a stove and a fridge" : "Call is no longer collecting replies"}
          disabled={!open}
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
