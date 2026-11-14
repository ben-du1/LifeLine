// How the dashboard words probabilities. A dispatcher reads "100%" as certain, but the model only ever has partial
// evidence (a room can lead just because it is the only photographed one of its type), so the extremes are never
// shown and the top room gets a confidence word next to its number.

export function pct(p) {
  const value = Math.round((p || 0) * 100);
  if (value >= 100) return ">99%";
  if (value <= 0) return "<1%";
  return `${value}%`;
}

// Signed change for the evidence line: "+32%", "−20%" (a real minus sign, so it lines up with the plus).
export function deltaPct(d) {
  const value = Math.max(1, Math.round(Math.abs(d) * 100));
  return `${d < 0 ? "−" : "+"}${value}%`;
}

// "Located" is reserved for the backend's decision (threshold, margin and floor all met); the rest is a reading of
// the top room's probability alone.
export function confidenceWord(status, probability) {
  if (status === "located") return { word: "Located", tone: "located" };
  if (probability >= 0.6) return { word: "Likely", tone: "likely" };
  if (probability >= 0.25) return { word: "Possible", tone: "possible" };
  return { word: "Unsure", tone: "unsure" };
}

// "→ Bathroom 1 +32%, Kitchen −20%": what one caller message did to the room probabilities.
export function describeEffect(entry) {
  if (!entry) return null;
  if (!entry.changes?.length) return "→ no change";
  return `→ ${entry.changes.map((c) => `${c.name} ${deltaPct(c.delta)}`).join(", ")}`;
}

// "Bathroom 2, then Bathroom 1": the order responders should search in, from the snapshot's sorted rooms.
export function checkOrder(rooms, count = 3) {
  const names = rooms.filter((r) => r.probability >= 0.05).slice(0, count).map((r) => r.name);
  return names.length ? names.join(", then ") : null;
}
