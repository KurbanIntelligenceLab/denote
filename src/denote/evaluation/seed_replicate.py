#!/usr/bin/env python3
"""Collect missing decoding seeds (default 1,2) into existing full_*.json files.

The normal ``full`` command skips any model with ``completed_at`` set, so seed
replication must clear that flag, collect only the requested seeds (resume-
safe: existing record_ids are skipped), then restore completion markers.

Usage:
  python denote_seed_replicate.py --dry-run
  python denote_seed_replicate.py --seeds 1,2 --limit 150 --skip-baselines
  python denote_seed_replicate.py --model qwen/qwen-2.5-7b-instruct --seeds 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from denote_v2_env import bootstrap

bootstrap()

from denote_experiments import (  # noqa: E402
    DEFAULT_MODELS,
    RAW,
    _acquire_model_lock,
    _model_run_complete,
    _release_model_lock,
    _slug,
    collect_interventional,
    load_arithmetic_problems,
    utc_now,
    write_json,
)
from denote_model import OpenRouterModel  # noqa: E402


def _seed_coverage(state: dict, seeds: list[int], n_problems: int) -> dict:
    """A seed is complete only when every panel problem has been attempted.

    Presence of *any* record with that seed is not enough — a crashed run can
    leave seed=1 on disk for 30/150 problems and still look \"done\".
    """
    problems_by_seed: dict[int, set] = {s: set() for s in seeds}
    for r in state.get("records", []):
        seed = r.get("seed")
        if seed is None:
            continue
        seed = int(seed)
        if seed in problems_by_seed:
            problems_by_seed[seed].add(r.get("problem_id"))
    have = sorted(s for s, pids in problems_by_seed.items() if len(pids) >= n_problems)
    missing = [s for s in seeds if len(problems_by_seed[s]) < n_problems]
    counts = {s: len(problems_by_seed[s]) for s in seeds}
    return {"have": have, "missing": missing, "counts": counts, "n_problems": n_problems}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--model", default="", help="Single model override")
    ap.add_argument("--skip-baselines", action="store_true", default=True)
    ap.add_argument("--include-baselines", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    models = [args.model] if args.model.strip() else [
        m.strip() for m in args.models.split(",") if m.strip()
    ]
    include_baselines = bool(args.include_baselines) and not args.skip_baselines
    # argparse default skip-baselines True; include-baselines flips it
    if args.include_baselines:
        include_baselines = True

    problems = load_arithmetic_problems(args.limit, include_math=True)
    print(f"problems={len(problems)} seeds={seeds} models={len(models)}")

    for model_name in models:
        path = RAW / f"full_{_slug(model_name)}.json"
        if not path.exists():
            print(f"[skip] no state file for {model_name}")
            continue
        state = json.loads(path.read_text(encoding="utf-8"))
        cov = _seed_coverage(state, seeds, n_problems=len(problems))
        print(
            f"{model_name:40s} counts={cov['counts']} missing={cov['missing']} "
            f"completed_at={bool(state.get('completed_at'))}",
            flush=True,
        )
        if not cov["missing"]:
            if not state.get("completed_at"):
                state["completed_at"] = utc_now()
                write_json(path, state)
                print(f"  marked completed_at (coverage full)", flush=True)
            else:
                print(f"  already complete for seeds {seeds}", flush=True)
            continue
        if args.dry_run:
            continue

        # Allow collect_interventional to run despite prior completion.
        state.pop("completed_at", None)
        state["on_policy_complete"] = state.get("on_policy_complete", False)
        write_json(path, state)

        lock = None
        try:
            lock = _acquire_model_lock(model_name)
            model = OpenRouterModel(model_name)
            # Pass all requested seeds that are incomplete; collector skips
            # existing record_ids so resumed runs only fill the gap.
            print(
                f"[seed-rep] collecting {model_name} seeds={cov['missing']} "
                f"(resume counts={cov['counts']})",
                flush=True,
            )
            collect_interventional(
                model,
                problems,
                seeds=cov["missing"],
                output_path=path,
                include_baselines=include_baselines,
            )
            state = json.loads(path.read_text(encoding="utf-8"))
            cov2 = _seed_coverage(state, seeds, n_problems=len(problems))
            if cov2["missing"]:
                raise RuntimeError(
                    f"still missing problem coverage after collect: {cov2['counts']}"
                )
            state["completed_at"] = utc_now()
            # Keep prior on-policy flag; seed replication does not re-collect E4.
            write_json(path, state)
            print(
                f"[seed-rep] finished {model_name} counts={cov2['counts']}",
                flush=True,
            )
        except Exception as exc:
            print(f"[seed-rep] FAILED {model_name}: {exc}", flush=True)
            return 1
        finally:
            if lock is not None:
                _release_model_lock(lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
