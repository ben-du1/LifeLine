import React from "react";

// Approach card under the plan: which side of the building and which floor the caller is on, in words responders
// arriving from the street can use, plus a Street View look at that wall. Built by the backend (approach.py).
export default function ApproachCard({ session }) {
  const card = session?.approach;
  // Only once the room is reasonably settled: a big "North side" card at 24% reads as a fact it isn't.
  const p = session?.estimate?.room_probability ?? 0;
  const shown = session?.status === "located" || session?.status === "best_guess" || p >= 0.5;
  if (!card || !shown) return null;
  const side = card.side
    ? `${card.street_relation === "front" || card.street_relation === "rear" ? `${capitalize(card.street_relation)}, ` : ""}${capitalize(card.side)} side`
    : "Side unknown";
  return (
    <div className={`approach-card${card.interior ? " interior" : ""}`}>
      <div className="approach-main">
        <div className="approach-big">
          <span>{side}</span>
          <span className="approach-floor">{card.multi_floor ? card.floor_text : card.floor_label}</span>
        </div>
        <p className="approach-text">{card.text}</p>
        <div className="approach-meta">
          {card.street && (
            <span className={card.faces_street ? "approach-tag front" : "approach-tag"}>
              {card.faces_street ? `Faces ${card.street}` : `Not on ${card.street}`}
            </span>
          )}
          {card.interior && <span className="approach-tag warn">Interior room</span>}
          {card.has_window_likely != null && !card.interior && (
            <span className="approach-tag">{card.has_window_likely ? "Window likely" : "Window unlikely"}</span>
          )}
          {card.streetview_url && (
            <a href={card.streetview_url} target="_blank" rel="noreferrer">
              Street View of this wall ↗
            </a>
          )}
        </div>
      </div>
      {card.streetview_image_url && (
        <a href={card.streetview_url} target="_blank" rel="noreferrer" className="approach-photo">
          <img src={card.streetview_image_url} alt={`Street View looking at the ${card.side} side`} />
        </a>
      )}
    </div>
  );
}

function capitalize(text) {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}
