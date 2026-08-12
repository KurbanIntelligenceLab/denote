#!/usr/bin/env python3
"""Re-call treat-arm continuations where answers.treat is None (no-response)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from denote_build import extract_answer
from denote_model import CONTINUE_SYSTEM, _CONTINUE_MAX_TOKENS
from denote_v2_env import bootstrap

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results" / "raw"
PILOT = ("qwen/qwen-2.5-7b-instruct", "2026-07-26")


def pending(state: dict) -> list[dict]:
    out = []
    for row in state.get("records", []):
        if row.get("status") != "built":
            continue
        if (row.get("answers") or {}).get("treat") is not None:
            continue
        if "treat" not in (row.get("conditions") or {}):
            continue
        out.append(row)
    return out


def _continue_max_tokens(model) -> int:
    return _CONTINUE_MAX_TOKENS.get(model.model, model.max_answer_tokens)


def _clear_empty_cache(model, question: str, trace: str) -> None:
    """Drop cached empty responses so reruns hit the live API."""
    max_tokens = _continue_max_tokens(model)
    token_caps = {max_tokens, model.max_answer_tokens}
    for cap in sorted(token_caps):
        cache_payload = {
            "model": model.model,
            "messages": [
                {"role": "system", "content": CONTINUE_SYSTEM},
                {"role": "user", "content": f"Question:\n{question}\n\nReasoning trace:\n{trace}"},
            ],
            "temperature": 0.0,
            "seed": 1,
            "max_tokens": cap,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cache_path = model.cache_dir / f"{cache_key}.json"
        if not cache_path.exists():
            continue
        try:
            record = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache_path.unlink(missing_ok=True)
            continue
        if not (record.get("text") or "").strip():
            cache_path.unlink(missing_ok=True)


def main() -> None:
    bootstrap()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="single model slug to rerun")
    ap.add_argument("--all", action="store_true", help="rerun all models with pending calls")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    total = 0
    targets: list[tuple[Path, dict]] = []
    for path in sorted(RAW.glob("full_*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        if (state.get("model"), state.get("access_date")) == PILOT:
            continue
        if args.model and state.get("model") != args.model:
            continue
        todo = pending(state)
        total += len(todo)
        if todo:
            targets.append((path, state))
            print(f"  {state['model']:38s} pending treat reruns: {len(todo)}")
    print(f"\ntotal pending: {total}")
    if args.dry_run or not (args.all or args.model):
        print("dry run only. Re-run with --all or --model <name> to execute.")
        return

    from denote_model import OpenRouterModel

    done_total = 0
    for path, state in targets:
        todo = pending(state)
        if not todo:
            continue
        model = OpenRouterModel(state["model"])
        done = 0
        for row in todo:
            trace = row["conditions"]["treat"]
            text = ""
            answer = None
            for attempt in range(3):
                try:
                    _clear_empty_cache(model, row["question"], trace)
                    text = model.continue_from(row["question"], trace)
                except Exception as exc:  # noqa: BLE001
                    print(f"    failed {row.get('record_id')}: {exc}", flush=True)
                    text = ""
                    break
                answer = extract_answer(text)
                if answer is not None and str(text).strip():
                    break
                if attempt < 2:
                    print(
                        f"    retry {row.get('record_id')} "
                        f"(empty={not str(text).strip()}, ans={answer})",
                        flush=True,
                    )
            if not str(text).strip() or answer is None:
                continue
            row.setdefault("outputs", {})["treat"] = text
            row.setdefault("answers", {})["treat"] = answer
            done += 1
            done_total += 1
            if done % 10 == 0:
                path.write_text(json.dumps(state, indent=1), encoding="utf-8")
                print(f"    {state['model']}: {done}/{len(todo)} checkpointed", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
        path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        print(f"  {state['model']}: wrote {done} treat answers -> {path.name}")

    print(f"\nfinished {done_total} reruns")


if __name__ == "__main__":
    main()
