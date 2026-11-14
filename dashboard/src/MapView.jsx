import React, { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./map.css";
import { getBuildingAerial, getBuildingGeo } from "./api.js";

// The floor plan in its real surroundings: the rooms of the shown floor (GeoJSON from /api/buildings/{id}/geo) on a
// street map, satellite tiles, or the building's Vexcel aerial photo. Matterport's geolocation can be a few meters
// off, which is why the first Vexcel overlay was removed; here the dispatcher can nudge and rotate the plan until it
// sits on the roof. The adjustment moves only our overlay (never the imagery) and is saved per model in
// localStorage, so it is done once per building. Plain Leaflet (no react-leaflet) keeps the bundle small; this file
// is lazy-loaded, so the default Plan view doesn't pay for it.

const METERS_PER_DEG_LAT = 111_320;
const NUDGE_M = 0.5;
const ROTATE_DEG = 1;
const NO_ALIGN = { dx: 0, dy: 0, rot: 0 };
// Tiles exist to z19; past that Leaflet upscales them, which is still useful for seeing a single house.
const MAX_ZOOM = 22;
const NATIVE_ZOOM = 19;

export const BASES = {
  streets: {
    label: "Streets",
    url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  },
  satellite: {
    label: "Satellite",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri &mdash; Esri, Maxar, Earthstar Geographics, and the GIS User Community",
  },
};

export default function MapView({ building, session, floor = 0, selectedRoom, onSelectRoom }) {
  const modelId = session.model_id;
  const [geo, setGeo] = useState(undefined); // undefined while loading, null when the model has no geolocation
  const [aerial, setAerial] = useState(null);
  const [error, setError] = useState("");
  const [base, setBase] = useState("satellite");
  const [align, setAlign] = useState(NO_ALIGN);
  const [adjusting, setAdjusting] = useState(false);
  const [map, setMap] = useState(null);
  const container = useRef(null);
  const layers = useRef({});
  // App passes a new callback every render; a ref keeps that from redrawing the overlay each time.
  const selectRoom = useRef(onSelectRoom);
  selectRoom.current = onSelectRoom;

  useEffect(() => {
    setGeo(undefined);
    setAerial(null);
    setError("");
    setAlign(loadAlign(modelId));
    getBuildingGeo(modelId).then(setGeo).catch((e) => (setGeo(null), setError(e.message)));
    // The Vexcel photo is optional (no key, or no coverage); the map works without it.
    getBuildingAerial(modelId)
      .then((a) => {
        setAerial(a);
        if (a) setBase("vexcel"); // the sharpest view of the roof, so it's the default when we have it
      })
      .catch(() => setAerial(null));
  }, [modelId]);

  // One Leaflet map per building, fitted to the building's footprint.
  useEffect(() => {
    if (!geo || !container.current) return;
    const m = L.map(container.current, { maxZoom: MAX_ZOOM, zoomSnap: 0.5 });
    m.createPane("aerial").style.zIndex = 250; // above the tiles, below the room overlay
    layers.current = { overlay: L.layerGroup().addTo(m) };
    fit(m, geo);
    setMap(m);
    return () => {
      m.remove();
      setMap(null);
    };
  }, [geo]);

  // Base layer. "Vexcel" is the satellite tiles with the building's aerial photo on top, so its edges blend in.
  useEffect(() => {
    if (!map) return;
    const tiles = BASES[base === "streets" ? "streets" : "satellite"];
    const group = L.layerGroup([
      L.tileLayer(tiles.url, { attribution: tiles.attribution, maxZoom: MAX_ZOOM, maxNativeZoom: NATIVE_ZOOM }),
    ]);
    if (base === "vexcel" && aerial) {
      group.addLayer(L.imageOverlay(aerial.url, aerial.bounds, {
        pane: "aerial",
        attribution: `Aerial imagery &copy; ${aerial.source || "Vexcel"}`,
      }));
    }
    group.addTo(map);
    return () => group.remove();
  }, [map, base, aerial]);

  const toLatLng = useMemo(() => geo && makeAligner(geo.center, align), [geo, align]);

  // Rooms, sweeps, the caller marker and the street label, redrawn whenever the session or alignment changes.
  useEffect(() => {
    const overlay = layers.current.overlay;
    if (!map || !overlay || !geo) return;
    overlay.clearLayers();
    const color = cssVar("--room", "#2f6fde");
    const accent = cssVar("--accent", "#c8102e");
    const probability = {};
    for (const room of session.rooms || []) probability[room.index] = room.probability;
    const estimate = session.estimate;
    const hasProbabilities = (session.rooms || []).length > 0;

    for (const feature of geo.features || []) {
      const props = feature.properties || {};
      if (props.kind !== "room" || (props.floor ?? 0) !== floor || feature.geometry?.type !== "Polygon") continue;
      const p = probability[props.index] ?? 0;
      const isTop = estimate?.room === props.index;
      const isSelected = selectedRoom === props.index;
      const polygon = L.polygon(feature.geometry.coordinates.map((ring) => ring.map(toLatLng)), {
        color: isSelected ? accent : color,
        weight: isSelected || isTop ? 2.5 : 1,
        dashArray: isSelected ? "6 4" : null,
        fillColor: color,
        // Same shading rule as FloorPlan, with a faint fill at 0% so every room stays clickable.
        fillOpacity: Math.max(0.06, Math.min(0.75, p * 1.2)),
      });
      const label = `${props.name}${hasProbabilities ? ` ${Math.round(p * 100)}%` : ""}`;
      // The top room's name stays on, just below the polygon's middle so it doesn't cover the caller marker.
      polygon.bindTooltip(escapeHtml(label), isTop
        ? { permanent: true, direction: "bottom", offset: [0, 16], className: "map-room-label" }
        : { direction: "center", className: "map-room-label" });
      polygon.on("click", () => selectRoom.current?.(props.index));
      overlay.addLayer(polygon);
    }

    (session.candidate_probabilities || []).forEach((p, i) => {
      const c = geo.candidates?.[i];
      if (!c || (c.floor ?? 0) !== floor) return;
      overlay.addLayer(L.circleMarker(toLatLng([c.lon, c.lat]), {
        radius: 2 + 7 * Math.sqrt(p), stroke: false, fillColor: "#fff", fillOpacity: 0.7, interactive: false,
      }));
    });

    const located = session.status === "located";
    const showEstimate = estimate && (estimate.floor ?? 0) === floor &&
      (located || session.status === "best_guess" || session.questions_asked > 0);
    const at = showEstimate && callerPosition(geo, estimate);
    if (at) {
      overlay.addLayer(L.marker(toLatLng(at), {
        interactive: false,
        keyboard: false,
        icon: L.divIcon({
          className: `map-marker ${located ? "located" : "tentative"}`,
          html: '<span class="map-marker-pulse"></span><span class="map-marker-dot"></span>',
          iconSize: [30, 30],
        }),
      }));
      // An uncertainty region from the estimate (model meters), drawn around the marker.
      if (estimate.region?.radius_m > 0) {
        overlay.addLayer(L.circle(toLatLng(at), {
          radius: estimate.region.radius_m, color: accent, weight: 1, dashArray: "4 4", fillOpacity: 0.08,
          interactive: false,
        }));
      }
    }

    const street = building?.orientation?.street;
    if (street) overlay.addLayer(streetLabel(geo.center, street));
  }, [map, geo, toLatLng, session, floor, selectedRoom, building]);

  function changeAlign(next) {
    setAlign(next);
    saveAlign(modelId, next);
  }
  const nudge = (de, dn, dr) =>
    changeAlign({ dx: round(align.dx + de), dy: round(align.dy + dn), rot: round(align.rot + dr) });

  if (geo === undefined) return <div className="plan-placeholder">Loading map…</div>;
  if (geo === null) {
    return (
      <div className="plan-placeholder">
        {error ? `Map unavailable: ${error}` : "Matterport has no geolocation for this building, so it can't be placed on a map."}
      </div>
    );
  }

  const center = toLatLng([geo.center.lon, geo.center.lat]);
  return (
    <div className="map-view">
      <div className="map-toolbar">
        <div className="segmented" role="group" aria-label="Base map">
          {Object.entries(BASES).map(([key, b]) => (
            <button key={key} type="button" className={base === key ? "active" : ""} onClick={() => setBase(key)}>
              {b.label}
            </button>
          ))}
          {aerial && (
            <button type="button" className={base === "vexcel" ? "active" : ""} onClick={() => setBase("vexcel")}
              title="Vexcel aerial photo of this building, 7.5 cm per pixel">
              Vexcel 7.5 cm
            </button>
          )}
        </div>
        <div className="map-links">
          <a href={googleMapsUrl(center)} target="_blank" rel="noreferrer">Google Maps ↗</a>
          <a href={streetViewUrl(geo.center, building?.orientation?.street)} target="_blank" rel="noreferrer">
            Street View ↗
          </a>
          <button type="button" className={adjusting ? "active" : ""} onClick={() => setAdjusting(!adjusting)}>
            Adjust alignment
          </button>
        </div>
      </div>
      <div className="map-frame">
        <div ref={container} className="map" />
        {adjusting && (
          <AlignControl
            align={align}
            onNudge={nudge}
            onReset={() => changeAlign(NO_ALIGN)}
            onFit={() => map && fit(map, geo)}
          />
        )}
      </div>
    </div>
  );
}

// Nudge arrows move the plan 0.5 m on the ground (not in screen pixels), so a step means the same at every zoom.
function AlignControl({ align, onNudge, onReset, onFit }) {
  const moved = align.dx || align.dy || align.rot;
  return (
    <div className="align-control" aria-label="Adjust plan alignment">
      <div className="align-pad">
        <button type="button" style={{ gridArea: "n" }} onClick={() => onNudge(0, NUDGE_M, 0)} title="Move north 0.5 m">↑</button>
        <button type="button" style={{ gridArea: "w" }} onClick={() => onNudge(-NUDGE_M, 0, 0)} title="Move west 0.5 m">←</button>
        <button type="button" style={{ gridArea: "e" }} onClick={() => onNudge(NUDGE_M, 0, 0)} title="Move east 0.5 m">→</button>
        <button type="button" style={{ gridArea: "s" }} onClick={() => onNudge(0, -NUDGE_M, 0)} title="Move south 0.5 m">↓</button>
        <button type="button" style={{ gridArea: "l" }} onClick={() => onNudge(0, 0, -ROTATE_DEG)} title="Rotate 1° counter-clockwise">⟲</button>
        <button type="button" style={{ gridArea: "r" }} onClick={() => onNudge(0, 0, ROTATE_DEG)} title="Rotate 1° clockwise">⟳</button>
      </div>
      <span className="muted align-readout">
        {moved ? `${fmt(align.dx, "E", "W")}, ${fmt(align.dy, "N", "S")}, ${align.rot}°` : "As geolocated"}
      </span>
      <div className="align-actions">
        <button type="button" onClick={onReset} disabled={!moved}>Reset</button>
        <button type="button" onClick={onFit}>Fit building</button>
      </div>
    </div>
  );
}

// [lon, lat] -> Leaflet [lat, lon], after rotating the plan clockwise by `rot` degrees around the building's center
// and shifting it dx meters east and dy meters north. A local flat-earth approximation is exact enough over a house.
function makeAligner(center, { dx, dy, rot }) {
  const mx = METERS_PER_DEG_LAT * Math.cos((center.lat * Math.PI) / 180);
  const t = (rot * Math.PI) / 180;
  const c = Math.cos(t);
  const s = Math.sin(t);
  return ([lon, lat]) => {
    const e = (lon - center.lon) * mx;
    const n = (lat - center.lat) * METERS_PER_DEG_LAT;
    const e2 = c * e + s * n + dx;
    const n2 = -s * e + c * n + dy;
    return [center.lat + n2 / METERS_PER_DEG_LAT, center.lon + e2 / mx];
  };
}

// [lon, lat] of the caller: the centroid of the estimate's region when it lists candidates, else the estimated
// sweep, else the middle of the estimated room.
function callerPosition(geo, estimate) {
  const candidates = geo.candidates || [];
  const byId = (id) => candidates.find((c) => c.candidate === id) ?? candidates[id];
  const ids = estimate.region?.candidates;
  if (Array.isArray(ids) && ids.length) {
    const found = ids.map(byId).filter(Boolean);
    if (found.length) {
      return [mean(found.map((c) => c.lon)), mean(found.map((c) => c.lat))];
    }
  }
  const c = estimate.candidate != null && byId(estimate.candidate);
  if (c) return [c.lon, c.lat];
  const room = (geo.features || []).find((f) => f.properties?.index === estimate.room);
  const ring = room?.geometry?.coordinates?.[0];
  return ring?.length ? [mean(ring.map((p) => p[0])), mean(ring.map((p) => p[1]))] : null;
}

// The nearest street (OpenStreetMap), labelled where it is and turned to run along the road. It is placed from the
// geolocated center, not the adjusted plan, because the road is real-world data like the base map.
function streetLabel(center, street) {
  const b = (street.bearing_deg * Math.PI) / 180;
  const mx = METERS_PER_DEG_LAT * Math.cos((center.lat * Math.PI) / 180);
  const at = [
    center.lat + (Math.cos(b) * street.distance_m) / METERS_PER_DEG_LAT,
    center.lon + (Math.sin(b) * street.distance_m) / mx,
  ];
  let angle = street.direction_deg - 90; // compass direction -> CSS rotation of horizontal text
  if (angle > 90) angle -= 180;
  if (angle < -90) angle += 180;
  return L.marker(at, {
    interactive: false,
    keyboard: false,
    icon: L.divIcon({
      className: "map-street",
      html: `<span style="transform: translate(-50%, -50%) rotate(${angle}deg)">${escapeHtml(street.name)} (nearest street)</span>`,
      iconSize: [0, 0],
    }),
  });
}

// Google needs no key for these deep links (Maps URLs API). Street View opens on the nearest road, facing the house.
function googleMapsUrl([lat, lon]) {
  return `https://www.google.com/maps/@?api=1&map_action=map&center=${lat.toFixed(7)},${lon.toFixed(7)}&zoom=20&basemap=satellite`;
}

function streetViewUrl(center, street) {
  if (!street) return `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${center.lat.toFixed(7)},${center.lon.toFixed(7)}`;
  const b = (street.bearing_deg * Math.PI) / 180;
  const lat = center.lat + (Math.cos(b) * street.distance_m) / METERS_PER_DEG_LAT;
  const lon = center.lon + (Math.sin(b) * street.distance_m) / (METERS_PER_DEG_LAT * Math.cos((center.lat * Math.PI) / 180));
  const heading = Math.round((street.bearing_deg + 180) % 360);
  return `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${lat.toFixed(7)},${lon.toFixed(7)}&heading=${heading}`;
}

function fit(map, geo) {
  if (geo.bounds) map.fitBounds(geo.bounds, { maxZoom: 21, padding: [24, 24] });
  else map.setView([geo.center.lat, geo.center.lon], 20);
}

// localStorage can throw (private mode, storage disabled); alignment then just isn't remembered.
const alignKey = (modelId) => `caller-locator:align:${modelId}`;
function loadAlign(modelId) {
  try {
    const saved = JSON.parse(localStorage.getItem(alignKey(modelId)));
    if (saved && [saved.dx, saved.dy, saved.rot].every(Number.isFinite)) return saved;
  } catch {
    // fall through
  }
  return NO_ALIGN;
}
function saveAlign(modelId, align) {
  try {
    if (align.dx || align.dy || align.rot) localStorage.setItem(alignKey(modelId), JSON.stringify(align));
    else localStorage.removeItem(alignKey(modelId));
  } catch {
    // not remembered; the adjustment still applies for this view
  }
}

function cssVar(name, fallback) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const round = (x) => Math.round(x * 10) / 10;
const fmt = (v, pos, neg) => `${Math.abs(v)} m ${v < 0 ? neg : pos}`;
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (ch) => `&#${ch.charCodeAt(0)};`);
