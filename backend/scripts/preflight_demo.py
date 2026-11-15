"""Demo preflight: make every demo building cached and known-good before judging, and say which ones are READY.

Usage (from backend/, on the laptop that has the .env keys):
    .venv/Scripts/python scripts/preflight_demo.py [--only TEXT] [--no-self-test] [--file demo_buildings.json]

For each entry of the repo-root demo_buildings.json:
  1. resolve the address with sessions.find_matches, exactly as a call would (exact match? score? listed model?);
  2. ingest the model if the cache is missing or stale (schema, embeddings): minutes per building, once;
  3. fetch north + the nearest street (orientation.ensure), and warm the aerial/geo caches if those modules exist;
  4. score it with compat.score_building if that module exists;
  5. self-test: run the entry's scripted caller messages through a real SessionManager and report the top room.
Then it prints a READY / NOT READY table with timings. Exit code 0 only if every building is READY.

Messaging is forced to "simulated" before anything is imported, so this never texts anyone, even though .env sets
MESSAGING_MODE=photon. The self-test uses a made-up 555 number as well.
"""

import argparse
import asyncio
import importlib
import os
import sys
import time
from pathlib import Path

os.environ["MESSAGING_MODE"] = "simulated"   # real environment variables win over .env (config._load_dotenv)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, demo, ingest, orientation  # noqa: E402
from app.sessions import SessionManager, find_matches  # noqa: E402

config.MESSAGING_MODE = "simulated"
TEST_PHONE = "+1 555 010 0000"
DEFAULT_SELF_TEST = ["I'm in a room with a window", "I don't know"]
# Optional cache-warming modules written by other parts of the project; each is used only if present.
WARMERS = (("aerial", ("ensure", "warm")), ("geo", ("ensure", "warm")))


class Check:
    """Results for one building: named step timings, problems that block READY, and warnings that don't."""

    def __init__(self, entry: dict):
        self.entry = entry
        self.model_id = entry.get("model_id")
        self.cells: dict[str, str] = {}
        self.times: dict[str, float] = {}
        self.problems: list[str] = []
        self.warnings: list[str] = []

    def timed(self, name: str, fn, *args):
        start = time.perf_counter()
        try:
            return fn(*args)
        finally:
            self.times[name] = time.perf_counter() - start

    @property
    def ready(self) -> bool:
        return not self.problems


def resolve(check: Check) -> bool:
    matches = check.timed("match", find_matches, check.entry["address"])
    if not matches:
        check.cells["match"] = "none"
        check.problems.append("no Matterport model matches this address")
        return False
    top = matches[0]
    check.cells["match"] = "exact" if top["exact"] else f"{round(top['score'] * 100)}%"
    if not top["exact"]:
        check.warnings.append(f"approximate match: {top['address']} ({round(top['score'] * 100)}%)")
    listed = check.entry.get("model_id")
    if listed and top["id"] != listed:
        # Fix demo_buildings.json (or the address) first; ingesting the wrong house would only waste minutes.
        check.problems.append(f"a call would open {top['id']} ({top['address']}), not the listed {listed}")
        return False
    check.model_id = top["id"]
    return True


def ensure_ingested(check: Check) -> bool:
    if ingest.is_ingested(check.model_id):
        check.cells["cache"] = "cached"
        return True
    print(f"  ingesting {check.model_id} (downloads + embeddings, takes minutes)...", flush=True)
    try:
        check.timed("ingest", ingest.ingest, check.model_id)
    except Exception as exc:  # UnusableModel, network or API errors: report and move on to the next building
        check.cells["cache"] = "FAILED"
        check.problems.append(f"ingest failed: {type(exc).__name__}: {exc}")
        return False
    check.cells["cache"] = "ingested"
    return True


def warm_extras(check: Check) -> None:
    found = check.timed("orientation", orientation.ensure, check.model_id)
    street = (found or {}).get("street") or {}
    check.cells["orient"] = street.get("name", "yes")[:18] if found else "none"
    if found is None:
        check.warnings.append("no orientation: the plan is drawn model-up, without the street")
    warmed = []
    for module_name, functions in WARMERS:
        try:
            module = importlib.import_module(f"app.{module_name}")
        except ImportError:
            continue
        fn = next((getattr(module, f) for f in functions if callable(getattr(module, f, None))), None)
        if fn is None:
            continue
        try:
            check.timed(module_name, fn, check.model_id)
            warmed.append(module_name)
        except Exception as exc:  # optional extras never block the demo
            check.warnings.append(f"{module_name} warm-up failed: {type(exc).__name__}: {exc}")
    check.cells["extras"] = "+".join(warmed) or "-"

    compat = check.timed("compat", demo.compat_score, check.model_id)
    check.cells["score"] = "-" if compat is None else f"{compat.get('score')}"
    check.warnings += [f"compat: {p}" for p in (compat or {}).get("problems") or []]


async def self_test(check: Check) -> None:
    """The entry's scripted caller messages through the real call flow (simulated messaging)."""
    async def notify(_):
        pass

    manager = SessionManager(notify)
    session = await manager.start_call(TEST_PHONE, check.entry["address"])
    while session.status == "loading":
        await asyncio.sleep(0.1)
    if session.status != "listening":
        check.cells["self-test"] = session.status
        check.problems.append(f"self-test call ended as {session.status}: {session.status_detail}")
        return
    for text in check.entry.get("self_test") or DEFAULT_SELF_TEST:
        await manager.handle_reply(session, text)
    snap = manager.snapshot(session)
    if session.status == "error" or not snap["rooms"]:
        check.cells["self-test"] = session.status
        check.problems.append(f"self-test failed: {session.status_detail}")
        return
    top = snap["rooms"][0]
    check.cells["self-test"] = f"{top['name'][:22]} {round(top['probability'] * 100)}%"
    expected = check.entry.get("expect_room")
    if expected and top["name"] != expected:
        check.warnings.append(f"self-test top room was {top['name']}, expected {expected}")


def run(check: Check, with_self_test: bool) -> None:
    print(f"- {check.entry.get('label') or check.entry['address']}", flush=True)
    if not resolve(check) or not ensure_ingested(check):
        return
    warm_extras(check)
    if with_self_test:
        start = time.perf_counter()
        try:
            asyncio.run(self_test(check))
        except Exception as exc:
            check.cells["self-test"] = "ERROR"
            check.problems.append(f"self-test raised {type(exc).__name__}: {exc}")
        check.times["self-test"] = time.perf_counter() - start


def print_table(checks: list[Check]) -> None:
    columns = ["building", "model", "match", "cache", "orient", "extras", "score", "self-test", "time", "verdict"]
    rows = []
    for c in checks:
        cells = {**c.cells, "building": (c.entry.get("label") or c.entry["address"])[:32],
                 "model": c.model_id or "-", "time": f"{sum(c.times.values()):.1f}s",
                 "verdict": "READY" if c.ready else "NOT READY"}
        rows.append([cells.get(col, "-") for col in columns])
    widths = [max(len(col), *(len(r[i]) for r in rows)) for i, col in enumerate(columns)]
    line = lambda values: "  ".join(v.ljust(w) for v, w in zip(values, widths))
    print("\n" + line(columns))
    print(line(["-" * w for w in widths]))
    for row in rows:
        print(line(row))
    for c in checks:
        notes = [f"NOT READY: {p}" for p in c.problems] + [f"warning: {w}" for w in c.warnings]
        if notes:
            print(f"\n{c.entry.get('label') or c.entry['address']}:")
            print("\n".join(f"  {n}" for n in notes))
        if c.times:
            print(f"  timings: {', '.join(f'{k} {v:.1f}s' for k, v in c.times.items())}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=demo.DEMO_FILE, help="demo building list (JSON)")
    parser.add_argument("--only", help="check only entries whose label or address contains this text")
    parser.add_argument("--no-self-test", action="store_true", help="skip the scripted call (faster)")
    args = parser.parse_args()
    entries = demo.load_entries(args.file)
    if args.only:
        needle = args.only.lower()
        entries = [e for e in entries if needle in f"{e.get('label', '')} {e['address']}".lower()]
    if not entries:
        print(f"No demo buildings in {args.file}")
        return 1
    print(f"Preflight for {len(entries)} demo building(s); messaging forced to {config.MESSAGING_MODE}.")
    checks = [Check(e) for e in entries]
    for check in checks:
        run(check, not args.no_self_test)
    print_table(checks)
    ready = sum(c.ready for c in checks)
    print(f"\n{ready}/{len(checks)} READY")
    return 0 if ready == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
