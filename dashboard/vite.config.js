import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dashboard talks to the FastAPI backend through Vite's proxy, so no CORS setup is needed.
// 127.0.0.1 rather than localhost: on macOS, Node resolves localhost to ::1 (IPv6) first, while uvicorn listens on
// IPv4 only, so the live-update socket could fail and the page only changed on reload.
const backend = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": backend,
      "/buildings": backend,
      "/ws": { target: backend.replace(/^http/, "ws"), ws: true },
    },
  },
});
