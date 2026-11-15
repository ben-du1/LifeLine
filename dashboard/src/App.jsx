import React, { useEffect, useMemo, useRef, useState } from "react";
import { askQuestion, getBuilding, getConfig, getRoomShots, getSessions, sendReply, startCall, subscribe } from "./api.js";
import FloorPlan from "./FloorPlan.jsx";

const DEMO_ADDRESS = "13122 Chipman Glen Dr, Houston, TX";
const STATUS_LABELS = {
  loading: "Loading building",
  listening: "Locating",
  located: "Located",
  best_guess: "Approximate",
  out: "Out of building",
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
        <dl className="topbar-meta">
          {config && (
            <>
              <dt>Texts</dt>
              <dd>{config.messaging_mode === "photon" ? "iMessage (Photon)" : "Simulated"}</dd>
              <dt>Replies read by</dt>
              <dd>{config.followups === "claude" ? "Claude" : "Keyword rules"}</dd>
            </>
          )}
          <dt>Server</dt>
          <dd className={connected ? "ok" : "warn"}>{connected ? "Connected" : "Reconnecting"}</dd>
        </dl>
      </header>

      <aside className="sidebar">
        <CallForm defaultPhone={config?.dev_test_phone || ""} onSubmit={handleCall} />
        {error && <p className="error">{error}</p>}
        <h2>Calls</h2>
        <ul className="session-list">
          {ordered.length === 0 && <li className="muted">No calls yet.</li>}
          {ordered.map((s) => (
            <li key={s.id}>
              <button
                className={s.id === selected?.id ? `session status-${s.status} active` : `session status-${s.status}`}
                onClick={() => setSelectedId(s.id)}
              >
                <span className="session-address">{s.address}</span>
                <span className="session-meta">
                  <span>{s.phone}</span>
                  <span>{clock(s.created)}</span>
                </span>
                <Status status={s.status} />
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main className="main">
        {selected ? (
          <SessionView session={selected} building={buildings[selected.model_id]} config={config} />
        ) : (
          <div className="empty">No active call. Enter the caller's phone and address to start.</div>
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
      <h2>New call</h2>
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

function Status({ status }) {
  return <span className={`status status-${status}`}>{STATUS_LABELS[status] || status}</span>;
}

// The one thing the dispatcher needs first: which room and floor, and how sure the system is.
function LocationSummary({ session, multiFloor }) {
  const estimate = session.estimate;
  const shown = estimate && (session.status === "located" || session.status === "best_guess" || session.questions_asked > 0);
  let where = "Not known yet";
  let label = "Caller location";
  if (session.status === "out") where = "Caller is out of the building";
  else if (shown) {
    where = multiFloor ? `${estimate.room_name}, ${estimate.floor_label}` : estimate.room_name;
    // Don't let a 47% guess read like a confirmed room.
    if (session.status === "best_guess") label = "Approximate location";
    else if (session.status !== "located") label = "Most likely so far (not confirmed)";
  }

  return (
    <section className={`summary status-${session.status}`}>
      <div className="summary-main">
        <span className="summary-label">{label}</span>
        <span className={["located", "out"].includes(session.status) ? "summary-where" : "summary-where tentative"}>{where}</span>
        <span className="summary-detail">
          <Status status={session.status} />
          {shown && session.status !== "out" && (
            <>
              {" "}· {pct(estimate.room_probability)} likely (next {pct(estimate.runner_up_probability)}) ·{" "}
              {estimate.checked ? "confirmed by a caller answer" : "not yet confirmed"} · position ±
              {estimate.radius_m?.toFixed(1)} m
            </>
          )}
        </span>
        {session.danger && <span className="danger">Caller reported smoke, fire, or trouble breathing</span>}
      </div>
      <dl className="summary-facts">
        <dt>Address</dt>
        <dd>{session.match?.address || session.address}</dd>
        <dt>Phone</dt>
        <dd>{session.phone}</dd>
        <dt>Questions sent</dt>
        <dd>
          {session.questions_asked} of {session.max_questions}
        </dd>
        <dt>Note</dt>
        <dd>{session.status_detail}</dd>
      </dl>
    </section>
  );
}

function SessionView({ session, building, config }) {
  const [showPhoto, setShowPhoto] = useState(false);
  // Room whose photos are shown; null follows the current best estimate.
  const [pickedRoom, setPickedRoom] = useState(null);
  // Floor shown on the plan; null follows the floor of the current best estimate.
  const [pickedFloor, setPickedFloor] = useState(null);
  // Scan point clicked on the plan; its photos are shown first. null when a room (not a dot) was picked.
  const [pickedSweep, setPickedSweep] = useState(null);
  useEffect(() => {
    setPickedRoom(null);
    setPickedFloor(null);
    setPickedSweep(null);
  }, [session.id]);
  const estimate = session.estimate;
  const shownRoom = pickedRoom ?? estimate?.room ?? null;
  const multiFloor = (building?.floors?.length || 0) > 1;
  const shownFloor = pickedFloor ?? estimate?.floor ?? 0;
  const floorplan = building?.floorplans?.[String(shownFloor)];

  function pickRoom(index) {
    setPickedRoom(index);
    setPickedSweep(null);
    const room = session.rooms.find((r) => r.index === index);
    if (room) setPickedFloor(room.floor); // show the floor the room is on
  }

  function pickSweep(candidate, roomIndex) {
    pickRoom(roomIndex);
    setPickedSweep(candidate);
  }

  return (
    <div className="session-view">
      <LocationSummary session={session} multiFloor={multiFloor} />

      <section className="panel plan-panel">
        <div className="panel-head">
          <div>
            <h2>Floor plan</h2>
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
            selectedSweep={pickedSweep}
            onSelectSweep={pickSweep}
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
            sweep={pickedSweep}
            followsEstimate={pickedRoom === null}
            onFollowEstimate={() => {
              setPickedRoom(null);
              setPickedSweep(null);
            }}
          />
        )}
      </section>

      <section className="panel conversation-panel">
        <h2>Texts with caller</h2>
        <Conversation messages={session.messages} />
        <DispatcherAsk session={session} />
        <CallerBox session={session} mode={config?.messaging_mode} />
      </section>

      <section className="panel rooms-panel">
        <h2>Rooms, most likely first</h2>
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
function RoomPhotos({ session, roomIndex, sweep, followsEstimate, onFollowEstimate }) {
  const [rooms, setRooms] = useState(null);
  const [error, setError] = useState("");
  const ready = session.rooms.length > 0;
  const clickedRef = useRef(null);
  // The gallery is below the plan, so bring a clicked point's photos into view.
  useEffect(() => {
    if (sweep !== null) clickedRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [sweep, rooms]);
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
  // A dot clicked on the plan: its photos go first. Numbers stay in the room's order, so "3 of 5" names the same
  // point before and after the click.
  const numbered = room.sweeps.map((s, i) => ({ ...s, number: i + 1 }));
  const clicked = numbered.find((s) => s.candidate === sweep);
  const sweeps = clicked ? [clicked, ...numbered.filter((s) => s !== clicked)] : numbered;

  return (
    <div className="views">
      <div className="views-head">
        <h3>
          Photos: {room.name} ({pct(live?.probability)})
          {room.area_m2 ? ` · ${room.area_m2} m²` : ""}
        </h3>
        {!followsEstimate && (
          <button type="button" className="link" onClick={onFollowEstimate}>
            Show most likely room
          </button>
        )}
      </div>
      {sweep !== null && !clicked && room.sweeps.length > 0 && (
        <p className="muted">No photos were taken at the scan point you clicked.</p>
      )}
      {room.sweeps.length === 0 ? (
        <p className="muted">No photos of this room. It can only be found from what the caller says.</p>
      ) : (
        sweeps.map((s) => (
          <div
            key={s.candidate}
            ref={s === clicked ? clickedRef : undefined}
            className={s === clicked ? "sweep-shots clicked" : "sweep-shots"}
          >
            <span className="muted">
              Scan point {s.number} of {room.sweeps.length}
              {s === clicked ? ` (clicked on the plan, ${pct(liveScan(session, s.candidate))})` : ""}
              {!clicked && s.number === 1 && followsEstimate && room.sweeps.length > 1
                ? `: caller most likely here (${pct(liveScan(session, s.candidate))})`
                : ""}
            </span>
            <div className="view-strip">
              {s.views.map((url) => (
                <a key={url} href={url} target="_blank" rel="noreferrer">
                  <img src={url} alt={`${room.name}, scan point ${s.number}`} loading="lazy" />
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
  if (match.exact) return <p className="match">{match.address} (exact address match)</p>;
  return (
    <p className="match match-approx" title={`Caller address: ${session.address}`}>
      Not an exact match. Using {match.address} ({pct(match.score)} similar to “{session.address}”). Confirm with the
      caller.
    </p>
  );
}

const WHO = { dispatch: "Sent", caller: "Caller", system: "Note" };

// The dispatcher keeps questioning at any point, including after the call is located: either the system's next best
// question (a check of the room, then whatever best separates the likely rooms, then where in the room) or their own.
function DispatcherAsk({ session }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");
  const open = ["listening", "located", "best_guess"].includes(session.status);
  useEffect(() => {
    setProblem("");
  }, [session.id]);

  async function ask(value) {
    setBusy(true);
    setProblem("");
    try {
      await askQuestion(session.id, value);
      setText("");
    } catch (e) {
      setProblem(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      className="dispatcher-ask"
      onSubmit={(e) => {
        e.preventDefault();
        if (text.trim()) ask(text);
      }}
    >
      <div className="dispatcher-row">
        <button type="button" className="primary" disabled={!open || busy} onClick={() => ask("")}>
          Ask next best question
        </button>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={open ? "Or type your own question to the caller" : "Call is not taking questions"}
          disabled={!open || busy}
          aria-label="Your own question to the caller"
        />
        <button type="submit" disabled={!open || busy || !text.trim()}>
          Send
        </button>
      </div>
      {problem && <p className="muted">{problem}</p>}
    </form>
  );
}

// Plain log, oldest first, with send/receive times, like a dispatch record.
function Conversation({ messages }) {
  const end = useRef(null);
  // Braces matter: current Chromium returns a Promise from scrollIntoView, and an effect that returns a value has it
  // called as a cleanup function, which crashed the log ("destroy is not a function") on every new message.
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [messages.length]);
  return (
    <ol className="messages">
      {messages.map((m, i) => (
        <li key={i} className={`msg msg-${m.role}`}>
          <span className="msg-time">{m.ts ? clock(m.ts) : ""}</span>
          <span className="msg-who">{m.source === "dispatcher" ? "You" : WHO[m.role] || m.role}</span>
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
  // A located call still takes replies ("I moved", "I'm out").
  const open = ["listening", "loading", "located", "best_guess"].includes(session.status);
  const quick = ["Yes", "No", "I don't know", "OUT"];

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
        Type a reply as the caller{mode === "photon" ? " (testing; real replies arrive by iMessage)" : ""}
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
// Backend timestamps are Unix seconds.
const clock = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
