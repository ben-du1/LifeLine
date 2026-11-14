import React, { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { getBuilding, getConfig, getRoomShots, getSessions, sendReply, startCall, subscribe } from "./api.js";
import { getAddresses, getDemoBuildings } from "./api.js";
import FloorPlan from "./FloorPlan.jsx";
import ApproachCard from "./ApproachCard.jsx";
import { checkOrder, confidenceWord, describeEffect, pct } from "./confidence.js";

// Leaflet is only needed once the dispatcher opens the map, so it is loaded then.
const MapView = lazy(() => import("./MapView.jsx"));
const AddressMap = lazy(() => import("./AddressMap.jsx"));

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

  async function handleCall(phone, address, modelId) {
    setError("");
    try {
      const session = await startCall(phone, address, modelId);
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
        <span className={`conn ${connected ? "on" : "off"}`}>{connected ? "Live" : "Reconnecting"}</span>
      </header>

      <aside className="sidebar">
        <CallForm defaultPhone={config?.dev_test_phone || ""} onSubmit={handleCall} />
        {error && <p className="error">{error}</p>}
        <h2 className="side-title">Calls</h2>
        <ul className="session-list">
          {ordered.length === 0 && <li className="muted">No calls yet.</li>}
          {ordered.map((s) => (
            <li key={s.id}>
              <button className={s.id === selected?.id ? "session active" : "session"} onClick={() => setSelectedId(s.id)}>
                <span className="session-address">{s.address}</span>
                <span className="session-meta">
                  <span>{s.phone}</span>
                  <StatusPill status={s.status} />
                </span>
              </button>
            </li>
          ))}
        </ul>
        {config && (
          <p className="side-foot">
            {config.messaging_mode === "photon" ? "Texts go out over iMessage (Photon)." : "Simulated caller: type replies in the call."}{" "}
            {config.followups === "claude" ? "Claude reads replies." : "Replies are read by keyword rules."}
          </p>
        )}
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
  // Demo buildings with their cache state (GET /api/demo, ready first). Empty if the backend has no list.
  const [demos, setDemos] = useState([]);
  useEffect(() => {
    if (!phone && defaultPhone) setPhone(defaultPhone);
  }, [defaultPhone]);
  useEffect(() => {
    getDemoBuildings().then(setDemos).catch(() => setDemos([]));
  }, []);
  const sameAddress = (a, b) => a.trim().toLowerCase() === b.trim().toLowerCase();
  const chosenDemo = demos.find((d) => sameAddress(d.address, address));
  // Suggestions from all ~10,000 callable addresses in the hackathon's model index (GET /api/addresses).
  const [suggestions, setSuggestions] = useState([]);
  useEffect(() => {
    const q = address.trim();
    if (q.length < 3) {
      setSuggestions([]);
      return undefined;
    }
    const timer = setTimeout(() => getAddresses(q).then(setSuggestions).catch(() => setSuggestions([])), 150);
    return () => clearTimeout(timer);
  }, [address]);
  const picked = suggestions.find((s) => sameAddress(s.address, address));

  return (
    <form
      className="call-form"
      onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        await onSubmit(phone, address, picked?.model_id || chosenDemo?.model_id || null);
        setBusy(false);
      }}
    >
      <h2 className="side-title">New 911 call</h2>
      <label>
        Caller phone
        <input value={phone} onChange={(e) => setPhone(e.target.value)} placeholder="+1 609 555 0100" required />
      </label>
      {demos.length > 0 && (
        <label>
          Demo buildings
          <select value={chosenDemo?.address || ""} onChange={(e) => e.target.value && setAddress(e.target.value)}>
            <option value="">Any address (type below)</option>
            {demos.map((d) => (
              <option key={d.address} value={d.address} title={[d.note, ...(d.problems || [])].join("\n")}>
                {d.ready ? "✓" : "⚠"} {d.label}
              </option>
            ))}
          </select>
        </label>
      )}
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
      {((chosenDemo && !chosenDemo.cached) || (picked && !picked.cached)) && (
        <p className="demo-warning">First call to this building downloads it, which takes a few minutes.</p>
      )}
      <button type="submit" className="primary" disabled={busy}>
        {busy ? "Starting…" : "Start call"}
      </button>
    </form>
  );
}

// The follow-up budget as a row of ticks: the questions really are a sequence, and the dispatcher sees at a glance
// how many are left before the system stops with its best guess.
function QuestionTicks({ asked, max }) {
  return (
    <p className="ticks" aria-label={`${asked} of ${max} follow-up questions asked`}>
      {Array.from({ length: max }, (_, i) => (
        <span key={i} className={i < asked ? "tick on" : "tick"} />
      ))}
      <span className="ticks-text">
        {asked} of {max} questions
      </span>
    </p>
  );
}

function capitalize(text) {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}

function StatusPill({ status }) {
  return <span className={`pill pill-${status}`}>{STATUS_LABELS[status] || status}</span>;
}

function SessionView({ session, building, config }) {
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
      <section className={`incident incident-${session.status}`}>
        <div className="incident-where">
          <StatusPill status={session.status} />
          {estimate ? (
            <p className="incident-room">
              {estimate.room_name}
              {multiFloor && <span className="incident-floor">{estimate.floor_label}</span>}
            </p>
          ) : (
            <p className="incident-room incident-none">
              {session.status === "listening" ? "No estimate yet" : session.status_detail}
            </p>
          )}
          {estimate?.zone_label && <p className="incident-zone">{capitalize(estimate.zone_label)}</p>}
          {!estimate && session.status === "listening" && (
            <p className="incident-zone">Waiting for the caller to describe the room.</p>
          )}
        </div>
        <div className="incident-numbers">
          {estimate && (
            <p className="incident-odds">
              <span className="incident-pct">{pct(estimate.room_probability)}</span>
              <ConfidenceTag status={session.status} probability={estimate.room_probability} />
              <span className="muted">next best {pct(estimate.runner_up_probability)}</span>
            </p>
          )}
          <QuestionTicks asked={session.questions_asked} max={session.max_questions} />
        </div>
        {session.status === "best_guess" && checkOrder(session.rooms) && (
          <p className="check-first">
            Search in this order: <strong>{checkOrder(session.rooms)}</strong>
          </p>
        )}
      </section>

      <section className="panel plan-panel">
        <div className="panel-head">
          <div>
            <h2>{building ? building.address : session.address}</h2>
            <MatchNote session={session} />
          </div>
          <div className="panel-tools">
            {floorplan && planMode === "plan" && (
              <label className="toggle">
                <input type="checkbox" checked={showPhoto} onChange={(e) => setShowPhoto(e.target.checked)} />
                Aerial photo
              </label>
            )}
            {building && (
              <div className="segmented" role="group" aria-label="Plan view">
                <button type="button" className={planMode === "plan" ? "active" : ""} onClick={() => setPlanMode("plan")}>
                  Plan
                </button>
                <button type="button" className={planMode === "map" ? "active" : ""} onClick={() => setPlanMode("map")}>
                  Map
                </button>
              </div>
            )}
          </div>
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
            {session.status === "no_model" || session.status === "error"
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
        <h2>Text thread with {session.phone}</h2>
        <Conversation messages={session.messages} evidence={session.evidence} />
        <CallerBox session={session} mode={config?.messaging_mode} />
      </section>

      <section className="panel rooms-panel">
        <h2>Where the caller could be</h2>
        {session.rooms.length === 0 ? (
          <p className="muted">Waiting for the building to load.</p>
        ) : !estimate && session.status === "listening" ? (
          <p className="muted">
            All {session.rooms.length} rooms are equally likely until the caller describes the room. The list fills in
            after their first useful reply.
          </p>
        ) : (
          <RoomBars session={session} shownRoom={shownRoom} multiFloor={multiFloor} onPick={pickRoom} />
        )}
      </section>
    </div>
  );
}

// The room list, most likely first. Rooms the backend can't tell apart yet (snapshot `tie_group`, when present) are
// bracketed together so the dispatcher doesn't read the order inside the group as meaningful.
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
    getRoomShots(session.id)
      .then((data) => setRooms(data.rooms))
      .catch((e) => setError(e.message));
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

// Which Matterport property the call is using. A non-exact match is highlighted so the dispatcher confirms it.
function MatchNote({ session }) {
  const match = session.match;
  if (!match) return null;
  if (match.exact) return <p className="match match-exact">Exact address match</p>;
  return (
    <p className="match match-approx" title={`Caller address: ${session.address}`}>
      Closest match to “{session.address}” ({pct(match.score)} similar). Confirm the address with the caller.
    </p>
  );
}

// Each caller message is followed by its effect on the room probabilities, from the snapshot's evidence log. Entries
// are matched by timestamp, then by text (each entry used once) in case a backend rounds the timestamp.
function Conversation({ messages, evidence = [] }) {
  const end = useRef(null);
  // A block body, not an expression: newer Chrome returns a value from scrollIntoView, and React called it as the
  // effect's clean-up, which crashed the page on every new message (blank until reloaded). "nearest" scrolls only
  // the conversation list, not the whole page.
  useEffect(() => {
    end.current?.scrollIntoView({ block: "nearest" });
  }, [messages.length, evidence.length]);
  const used = new Set();
  const effectFor = (m) => {
    let entry = evidence.find((e, i) => !used.has(i) && e.ts === m.ts && e.text === m.text);
    entry ??= evidence.find((e, i) => !used.has(i) && e.text === m.text);
    if (entry) used.add(evidence.indexOf(entry));
    return entry;
  };
  return (
    <ol className="messages">
      {messages.map((m, i) => {
        const entry = m.role === "caller" ? effectFor(m) : null;
        return (
          <li key={i} className={`msg msg-${m.role}`}>
            <span className="visually-hidden">{m.role === "dispatch" ? "Sent to caller:" : m.role === "caller" ? "Caller:" : "Note:"}</span>
            <span className="msg-text">{m.text}</span>
            {entry && (
              <span className="msg-effect" title={`Top room before this message: ${entry.top_before}`}>
                {describeEffect(entry)}
              </span>
            )}
          </li>
        );
      })}
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
      <label htmlFor="caller-input">
        Simulated caller reply{mode === "photon" ? " (overrides iMessage for testing)" : ""}
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
