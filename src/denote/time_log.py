#!/usr/bin/env python3
"""Append start/stop/event rows to results/v2_time_log.jsonl.

Usage:
  python denote_time_log.py start TASK_ID [--note "..."]
  python denote_time_log.py stop  TASK_ID [--note "..."] [--outcome "..."]
  python denote_time_log.py event TASK_ID [--note "..."] [--outcome "..."]
  python denote_time_log.py summary
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "results" / "v2_time_log.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _append(row: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load() -> list[dict]:
    if not LOG.exists():
        return []
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def cmd_start(task_id: str, note: str, outcome: str) -> None:
    _append(
        {
            "ts": _now(),
            "event": "start",
            "task_id": task_id,
            "note": note or "",
            "outcome": outcome or "",
        }
    )
    print(f"START {task_id} @ {_now()}")


def cmd_stop(task_id: str, note: str, outcome: str) -> None:
    rows = _load()
    start_ts = None
    for row in reversed(rows):
        if row.get("task_id") == task_id and row.get("event") == "start":
            start_ts = row["ts"]
            break
    duration_s = None
    if start_ts:
        t0 = datetime.fromisoformat(start_ts)
        t1 = datetime.fromisoformat(_now())
        duration_s = round((t1 - t0).total_seconds(), 1)
    _append(
        {
            "ts": _now(),
            "event": "stop",
            "task_id": task_id,
            "note": note or "",
            "outcome": outcome or "",
            "duration_s": duration_s,
            "start_ts": start_ts,
        }
    )
    dur = f"{duration_s}s" if duration_s is not None else "unknown"
    print(f"STOP  {task_id} @ {_now()}  duration={dur}  outcome={outcome or '-'}")


def cmd_event(task_id: str, note: str, outcome: str) -> None:
    _append(
        {
            "ts": _now(),
            "event": "event",
            "task_id": task_id,
            "note": note or "",
            "outcome": outcome or "",
        }
    )
    print(f"EVENT {task_id} @ {_now()}  {note or outcome or ''}")


def cmd_summary() -> None:
    rows = _load()
    if not rows:
        print(f"no log at {LOG}")
        return
    print(f"{'task_id':28s} {'event':6s} {'duration_s':>10s}  ts")
    for row in rows:
        dur = row.get("duration_s")
        dur_s = f"{dur:.1f}" if isinstance(dur, (int, float)) else "-"
        print(f"{row.get('task_id',''):28s} {row.get('event',''):6s} {dur_s:>10s}  {row.get('ts','')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["start", "stop", "event", "summary"])
    ap.add_argument("task_id", nargs="?", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--outcome", default="")
    args = ap.parse_args()
    if args.action == "summary":
        cmd_summary()
        return
    if not args.task_id:
        sys.exit("task_id required for start/stop/event")
    if args.action == "start":
        cmd_start(args.task_id, args.note, args.outcome)
    elif args.action == "stop":
        cmd_stop(args.task_id, args.note, args.outcome)
    else:
        cmd_event(args.task_id, args.note, args.outcome)


if __name__ == "__main__":
    main()
