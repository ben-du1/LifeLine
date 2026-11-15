import React, { useEffect, useMemo, useRef, useState } from "react";
import { pct } from "./confidence.js";

// Draws the building in meters. When the backend knows the building's heading (building.orientation), the plan is
// rotated north-up and the nearest street is drawn on the side where it really is, so responders can match the plan
// to what they see from the road. Coordinates are rotated here (not with an SVG group transform) so text stays
// upright. SVG y grows downward and model y grows upward, so y is negated.
// The schematic comes from room wall segments because Matterport's raster floorplans are aerial photos of the roof.
// Only `floor` is drawn; the view box covers every floor so the scale doesn't jump when switching floors.
// "Zoom to room" narrows the view box to the estimated room so a large living room fills the panel; it follows the
// estimate as it moves. estimate.region (where in the room, from the in-room sweeps) is optional and drawn when present.
export default function FloorPlan({ building, session, floor = 0, showPhoto, selectedRoom, onSelectRoom }) {
  const orientation = building.orientation;
  const yawDeg = orientation?.yaw_deg ?? 0;
  const P = useMemo(() => makeProjector(yawDeg), [yawDeg]);
  const bounds = useMemo(() => computeBounds(building, P), [building, P]);
  const roomProbability = useMemo(() => {
    const byIndex = {};
    for (const room of session?.rooms || []) byIndex[room.index] = room.probability;
    return byIndex;
  }, [session]);

  const estimate = session?.estimate;
  const located = session?.status === "located";
  const showEstimate = estimate && (estimate.floor ?? 0) === floor &&
    (located || session?.status === "best_guess" || session?.questions_asked > 0);
  const fp = building.floorplans?.[String(floor)];
  const onFloor = (room) => (room.floor ?? 0) === floor;
  const roomIndex = useMemo(() => Object.fromEntries(building.rooms.map((r, i) => [r.id, i])), [building]);
  const street = orientation?.street;
  // Room around the plan for the compass and the street band.
  const pad = orientation ? 4.5 : 1;
  const fullView = {
    x: bounds.minX - pad,
    y: bounds.minY - pad,
    w: bounds.maxX - bounds.minX + 2 * pad,
    h: bounds.maxY - bounds.minY + 2 * pad,
  };
  const region = showEstimate ? estimate.region : null;
  const markerAt = region ? P(region.x, region.y) : showEstimate ? P(estimate.x, estimate.y) : null;
  const estimateRoom = showEstimate ? building.rooms[estimate.room] : null;
  const roomView = estimateRoom && onFloor(estimateRoom) ? roomViewBox(estimateRoom, region, P, fullView) : null;
  const [zoomed, setZoomed] = useState(false);
  const view = useAnimatedView(zoomed && roomView ? roomView : fullView);
  // Compass and marker are sized in meters; shrink them with the zoom so they don't swamp a zoomed-in room.
  const scale = Math.max(0.35, Math.min(1, view.w / fullView.w));
  const tied = new Set((session?.tie_group?.length > 1 ? session.tie_group : []).map((t) => t.index));
  const zoneLabel = showEstimate ? estimate.zone_label : null;

  return (
    <>
    {roomView && (
      <div className="plan-tools">
        <button type="button" className="zoom-toggle" aria-pressed={zoomed} onClick={() => setZoomed(!zoomed)}>
          {zoomed ? "Show whole floor" : `Zoom to ${estimate.room_name || "room"}`}
        </button>
      </div>
    )}
    <div className="floorplan-viewer">
    <svg
      className="floorplan"
      viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
      role="img"
      aria-label="Floor plan with predicted caller position"
    >
      {showPhoto && fp && (
        <g transform={`rotate(${-yawDeg})`}>
          <image
            href={fp.url}
            x={fp.origin.x}
            y={-(fp.origin.y + fp.height / fp.resolution)}
            width={fp.width / fp.resolution}
            height={fp.height / fp.resolution}
            opacity="0.45"
          />
        </g>
      )}
      {street && <StreetBand street={street} bounds={bounds} />}
      {building.rooms.map((room, index) => {
        if (!onFloor(room)) return null;
        const p = roomProbability[index] ?? 0;
        const isTop = estimate?.room === index;
        return (
          <g key={room.id}>
            {room.polygon && (
              <polygon
                points={room.polygon.map(([x, y]) => P(x, y).join(",")).join(" ")}
                className="room-fill"
                // Keep a faint fill even at 0% so every room is clickable for its photos.
                style={{ fillOpacity: Math.max(0.04, Math.min(0.85, p * 1.2)) }}
                data-top={isTop || undefined}
                data-selected={selectedRoom === index || undefined}
                data-tied={tied.has(index) || undefined}
                onClick={() => onSelectRoom?.(index)}
              >
                <title>{`${room.name}: click to see its photos`}</title>
              </polygon>
            )}
            {room.walls.map(([x1, y1, x2, y2], i) => {
              const [ax, ay] = P(x1, y1);
              const [bx, by] = P(x2, y2);
              return <line key={i} x1={ax} y1={ay} x2={bx} y2={by} className="wall" />;
            })}
          </g>
        );
      })}
      {building.rooms.map((room, index) => {
        if (!room.centroid || !onFloor(room)) return null;
        const [cx, cy] = P(room.centroid[0], room.centroid[1]);
        // Closets and other small rooms crowd their neighbours' labels: name them only when they matter (likely,
        // estimated or selected). Their outline and the hover title still identify them.
        const small = polygonArea(room.polygon) < 4;
        const matters = index === estimate?.room || index === selectedRoom || (roomProbability[index] ?? 0) >= 0.1;
        if (small && !(showEstimate && matters)) return null;
        return (
          <text key={room.id} x={cx} y={cy} className="room-label">
            <tspan x={cx}>{room.name}</tspan>
            {showEstimate && session?.rooms?.length > 0 && (
              <tspan x={cx} dy="0.45" className="room-prob">
                {pct(roomProbability[index] ?? 0)}
              </tspan>
            )}
          </text>
        );
      })}
      {(session?.candidate_probabilities || []).map((p, i) => {
        const c = building.candidates[i];
        if (!onFloor(building.rooms[roomIndex[c.room_id]])) return null;
        const [cx, cy] = P(c.x, c.y);
        return <circle key={c.id} cx={cx} cy={cy} r={0.12 + 0.35 * Math.sqrt(p)} className="sweep" />;
      })}
      {region && <Region region={region} P={P} />}
      {markerAt && <Marker at={markerAt} located={located} scale={scale} />}
    </svg>
      {orientation && <Compass />}
    </div>
    {zoneLabel && (
      <p className="plan-zone">
        Likely spot: <strong>{zoneLabel}</strong>
      </p>
    )}
    </>
  );
}

// Where in the room: a soft circle of radius_m around the region centroid, plus one heat dot per in-room sweep
// sized and shaded by its share of the room's probability. A radius is unchanged by the north-up rotation.
function Region({ region, P }) {
  const [cx, cy] = P(region.x, region.y);
  const sweeps = region.sweeps || [];
  const maxP = Math.max(1e-9, ...sweeps.map((s) => s.p));
  return (
    <g className="region">
      {region.radius_m > 0 && <circle cx={cx} cy={cy} r={Math.max(0.4, region.radius_m)} className="region-area" />}
      {sweeps.map((s) => {
        const [x, y] = P(s.x, s.y);
        const share = s.p / maxP;
        return (
          <circle
            key={s.candidate}
            cx={x}
            cy={y}
            r={0.12 + 0.3 * Math.sqrt(share)}
            className="region-sweep"
            style={{ fillOpacity: 0.2 + 0.7 * share }}
          />
        );
      })}
    </g>
  );
}

// View box around one room (walls, polygon and the region circle), padded, at least MIN_ZOOM_M across so a closet
// doesn't fill the panel, and never larger than the whole-floor view.
const MIN_ZOOM_M = 7;
function roomViewBox(room, region, P, fullView) {
  const pts = [];
  for (const [x1, y1, x2, y2] of room.walls) pts.push(P(x1, y1), P(x2, y2));
  for (const [x, y] of room.polygon || []) pts.push(P(x, y));
  if (region) {
    const [cx, cy] = P(region.x, region.y);
    const r = region.radius_m || 0;
    pts.push([cx - r, cy - r], [cx + r, cy + r]);
  }
  if (!pts.length) return null;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const pad = 1.2;
  let w = Math.max(MIN_ZOOM_M, Math.max(...xs) - Math.min(...xs) + 2 * pad);
  let h = Math.max(MIN_ZOOM_M, Math.max(...ys) - Math.min(...ys) + 2 * pad);
  // Keep the whole-floor aspect ratio so the panel's height doesn't jump when zooming.
  const aspect = fullView.w / fullView.h;
  if (w / h < aspect) w = h * aspect;
  else h = w / aspect;
  if (w >= fullView.w) return null; // the room already fills the plan
  const mx = (Math.max(...xs) + Math.min(...xs)) / 2;
  const my = (Math.max(...ys) + Math.min(...ys)) / 2;
  return { x: mx - w / 2, y: my - h / 2, w, h };
}

// Eases the view box toward its target (SVG viewBox can't be CSS-transitioned). Jumps straight there when the user
// prefers reduced motion.
function useAnimatedView(target) {
  const [view, setView] = useState(target);
  const current = useRef(target);
  const key = `${target.x} ${target.y} ${target.w} ${target.h}`;
  useEffect(() => {
    const from = current.current;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      current.current = target;
      setView(target);
      return;
    }
    const start = performance.now();
    const duration = 350;
    let frame;
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const e = 1 - (1 - t) ** 3;
      const next = {
        x: from.x + (target.x - from.x) * e,
        y: from.y + (target.y - from.y) * e,
        w: from.w + (target.w - from.w) * e,
        h: from.h + (target.h - from.h) * e,
      };
      current.current = next;
      setView(next);
      if (t < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [key]);
  return view;
}

function Marker({ at: [x, y], located, scale = 1 }) {
  return (
    <g className={located ? "marker located" : "marker tentative"}>
      <circle cx={x} cy={y} r={0.9 * scale} className="marker-pulse" />
      <circle cx={x} cy={y} r={0.35 * scale} className="marker-dot" />
    </g>
  );
}

// North arrow. The plan is drawn north-up whenever this is shown.
function Compass() {
  return (
    <svg className="compass" viewBox="0 0 44 44" role="img" aria-label="North is up">
      <circle cx="22" cy="22" r="21.25" />
      <path d="M22,10 L29,34 L22,29 L15,34 Z" />
      <text x="22" y="-5">N</text>
    </svg>
  );
}

// The nearest street (OpenStreetMap), drawn as a band just outside the plan on the side where it is, running in the
// road's real direction. Its distance in meters is from the middle of the house to the road's centerline.
function StreetBand({ street, bounds }) {
  const b = (street.bearing_deg * Math.PI) / 180;
  const toward = [Math.sin(b), -Math.cos(b)]; // compass bearing -> SVG direction (north-up)
  const cx = (bounds.minX + bounds.maxX) / 2;
  const cy = (bounds.minY + bounds.maxY) / 2;
  const hx = (bounds.maxX - bounds.minX) / 2;
  const hy = (bounds.maxY - bounds.minY) / 2;
  // Distance from the plan's middle to its edge in that direction, then a little beyond.
  const edge = Math.min(Math.abs(toward[0]) > 1e-6 ? hx / Math.abs(toward[0]) : Infinity,
                        Math.abs(toward[1]) > 1e-6 ? hy / Math.abs(toward[1]) : Infinity);
  const at = [cx + toward[0] * (edge + 2.2), cy + toward[1] * (edge + 2.2)];
  const d = (street.direction_deg * Math.PI) / 180;
  const along = [Math.sin(d), -Math.cos(d)];
  const half = Math.max(hx, hy) + 3;
  // Keep the label reading left-to-right.
  let angle = (Math.atan2(along[1], along[0]) * 180) / Math.PI;
  if (angle > 90) angle -= 180;
  if (angle < -90) angle += 180;
  return (
    <g className="street">
      <line
        x1={at[0] - along[0] * half}
        y1={at[1] - along[1] * half}
        x2={at[0] + along[0] * half}
        y2={at[1] + along[1] * half}
        className="street-band"
      />
      <text x={at[0]} y={at[1]} transform={`rotate(${angle} ${at[0]} ${at[1]})`} className="street-label">
        {street.name}, nearest street, {Math.round(street.distance_m)} m {street.side}
      </text>
    </g>
  );
}

// Model meters -> SVG coordinates, rotated north-up by the model's heading (counter-clockwise yaw).
function makeProjector(yawDeg) {
  const t = (yawDeg * Math.PI) / 180;
  const c = Math.cos(t);
  const s = Math.sin(t);
  return (x, y) => [c * x - s * y, -(s * x + c * y)];
}

function computeBounds(building, P) {
  const xs = [];
  const ys = [];
  const add = ([x, y]) => (xs.push(x), ys.push(y));
  for (const room of building.rooms) {
    for (const [x1, y1, x2, y2] of room.walls) add(P(x1, y1)), add(P(x2, y2));
  }
  for (const c of building.candidates) add(P(c.x, c.y));
  if (!xs.length) return { minX: -5, maxX: 5, minY: -5, maxY: 5 };
  return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
}

// Floor area of a room outline in m² (shoelace); 0 when there is no outline.
function polygonArea(polygon) {
  if (!polygon || polygon.length < 3) return 0;
  let sum = 0;
  for (let i = 0; i < polygon.length; i++) {
    const [x1, y1] = polygon[i];
    const [x2, y2] = polygon[(i + 1) % polygon.length];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) / 2;
}
