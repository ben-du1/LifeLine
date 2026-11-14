import React, { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { BASES } from "./MapView.jsx";

// Where the typed address is, before the call starts: a pin for each suggested building, and once one is picked,
// a close satellite view of that lot so the dispatcher can check it is the right house. Clicking a pin picks it.
export default function AddressMap({ suggestions, picked, onPick }) {
  const box = useRef(null);
  const map = useRef(null);
  const layers = useRef({ base: null, pins: null, mode: null });

  useEffect(() => {
    map.current = L.map(box.current, { zoomControl: false, attributionControl: true, maxZoom: 21 });
    map.current.setView([31.5, -106.5], 4); // between Houston and Los Angeles until there is something to show
    layers.current.pins = L.layerGroup().addTo(map.current);
    return () => map.current.remove();
  }, []);

  useEffect(() => {
    const m = map.current;
    const places = (picked ? [picked] : suggestions).filter((s) => s.lat != null && s.lon != null);
    // Streets to tell several matches apart; satellite once one building is chosen, to see the actual roof.
    const mode = picked ? "satellite" : "streets";
    if (layers.current.mode !== mode) {
      layers.current.base?.remove();
      const base = BASES[mode];
      layers.current.base = L.tileLayer(base.url, { attribution: base.attribution, maxZoom: 21, maxNativeZoom: 19 }).addTo(m);
      layers.current.mode = mode;
    }
    layers.current.pins.clearLayers();
    places.forEach((s) => {
      const pin = L.circleMarker([s.lat, s.lon], {
        radius: picked ? 9 : 6,
        color: "#fff",
        weight: 2,
        fillColor: picked ? "#c41e2c" : "#2a5c8a",
        fillOpacity: 1,
      });
      pin.bindTooltip(s.address, { direction: "top", offset: [0, -6] });
      if (!picked) pin.on("click", () => onPick(s));
      pin.addTo(layers.current.pins);
    });
    if (places.length === 1) m.setView([places[0].lat, places[0].lon], picked ? 19 : 16);
    else if (places.length > 1) m.fitBounds(L.latLngBounds(places.map((s) => [s.lat, s.lon])), { padding: [24, 24], maxZoom: 16 });
  }, [suggestions, picked]);

  return <div ref={box} className="address-map" role="img" aria-label="Map of the typed address" />;
}
