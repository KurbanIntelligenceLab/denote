#!/usr/bin/env python3
"""denote_legacy_score.py -- score the legacy arm, which was built and never run.

Every raw record already carries conditions['legacy']: the corruption prompt in
the style of Lanham et al. was constructed for all 1,306 items. No record has
answers['legacy'], because the model was never called on it. The paper's own
head-to-head therefore does not exist.

This resumes exactly that: it calls the model on the legacy trace, extracts an
answer with the same extractor the rest of the pipeline uses, and writes the
result back into the raw state so analyze_state picks it up.

    python denote_legacy_score.py --dry-run     # count what would be called
    python denote_legacy_score.py --model anthropic/claude-sonnet-5
    python denote_legacy_score.py --all

Needs OPENROUTER_API_KEY, like the original run. Resumable: an item that
already has answers['legacy'] is skipped, so it can be interrupted.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

from denote_v2_env import bootstrap

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results" / "raw"
PILOT = ("qwen/qwen-2.5-7b-instruct", "2026-07-26")


def states(only=None, exclude_grammar=False):
    for p in sorted(RAW.glob("full_*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        if (s.get("model"), s.get("access_date")) == PILOT:
            continue
        if only and s.get("model") != only:
            continue
        if exclude_grammar and "_grammar" in p.stem:
            continue
        yield p, s


def pending(state):
    out = []
    for r in state.get("records", []):
        if r.get("status") != "built":
            continue
        conds = r.get("conditions") or {}
        if "legacy" not in conds:
            continue
        if (r.get("answers") or {}).get("legacy") is not None:
            continue
        out.append(r)
    return out


def main():
    bootstrap()
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--exclude-grammar", action="store_true",
                    help="skip *_grammar.json files even when --model matches "
                         "their model field (grammar variants share the same "
                         "model name as the canonical file)")
    a = ap.parse_args()

    total = 0
    for p, s in states(a.model, a.exclude_grammar):
        todo = pending(s)
        total += len(todo)
        print(f"  {s['model']:38s} ({p.name}) legacy calls pending: {len(todo)}")
    print(f"\ntotal pending: {total}")
    if a.dry_run or not (a.model or a.all):
        print("dry run only. Re-run with --all or --model <name> to execute.")
        return

    from denote_model import OpenRouterModel          # same client as the run
    from denote_build import extract_answer           # same extractor

    for p, s in states(a.model, a.exclude_grammar):
        todo = pending(s)
        if not todo:
            continue
        mdl = OpenRouterModel(s["model"])
        done = 0
        for r in todo:
            trace = r["conditions"]["legacy"]
            try:
                text = mdl.continue_from(r["question"], trace)
            except Exception as exc:                  # noqa: BLE001
                print(f"    call failed on {r.get('record_id')}: {exc}")
                continue
            r.setdefault("outputs", {})["legacy"] = text
            r.setdefault("answers", {})["legacy"] = extract_answer(text)
            done += 1
            if done % 25 == 0:
                p.write_text(json.dumps(s, indent=1), encoding="utf-8")
                print(f"    {s['model']}: {done}/{len(todo)} (checkpointed)")
            if a.sleep:
                time.sleep(a.sleep)
        p.write_text(json.dumps(s, indent=1), encoding="utf-8")
        print(f"  {s['model']}: wrote {done} legacy answers -> {p.name}")

    print("\nNow re-run the analysis so the legacy column appears:")
    print("  python denote_experiments.py --analyze     (or analyze_all())")
    print("  python denote_reconcile.py")
    print("Then fill the legacy head-to-head worksheet in the paper appendix.")


if __name__ == "__main__":
    main()
