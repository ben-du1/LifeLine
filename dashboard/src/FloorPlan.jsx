import React, { useMemo } from "react";

// Draws the building in meters. When the backend knows the building's heading (building.orientation), the plan is
// rotated north-up and the nearest street is drawn on the side where it really is, so responders can match the plan
// to what they see from the road. Coordinates are rotated here (not with an SVG group transform) so text stays
// upright. SVG y grows downward and model y grows upward, so y is negated.
// The schematic comes from room wall segments because Matterport's raster floorplans are aerial photos of the roof.
// Only `floor` is drawn; the view box covers every floor so the scale doesn't jump when switching floors.
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
  const view = {
    x: bounds.minX - pad,
    y: bounds.minY - pad,
    w: bounds.maxX - bounds.minX + 2 * pad,
    h: bounds.maxY - bounds.minY + 2 * pad,
  };

  return (
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
          return (
            <text key={room.id} x={cx} y={cy} className="room-label">
              <tspan x={cx}>{room.name}</tspan>
              {session?.rooms?.length > 0 && (
                <tspan x={cx} dy="0.45" className="room-prob">
                  {Math.round((roomProbability[index] ?? 0) * 100)}%
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
        {showEstimate && <Marker at={P(estimate.x, estimate.y)} radius={estimate.radius_m} located={located} />}
      </svg>
      {orientation && <Compass />}
    </div>
  );
}

// The caller's estimated spot, with a circle for how sure the system is about where in the room (RMS spread of the
// room's scan points, in meters). It shrinks as "what's closest to you?" answers come in.
function Marker({ at: [x, y], radius, located }) {
  return (
    <g className={located ? "marker located" : "marker tentative"}>
      {radius > 0 && <circle cx={x} cy={y} r={radius} className="marker-area" />}
      <circle cx={x} cy={y} r="0.9" className="marker-pulse" />
      <circle cx={x} cy={y} r="0.35" className="marker-dot" />
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
        {street.name} · nearest street · {Math.round(street.distance_m)} m {street.side}
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
