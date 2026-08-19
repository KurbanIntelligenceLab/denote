#!/usr/bin/env python3
"""V1 parser spot-check: sample parsed traces for human agreement.

  phase 1   python denote_v1_parser_audit.py --sample --n 100
            Writes results/v1_parser_audit_sheet.csv

  phase 2   Human fills:
              human_denotation  — what the trace computes (number)
              agree             — yes | no  (vs parser_den_clean)
            optional note

  phase 3   python denote_v1_parser_audit.py --analyze
            Writes results/v1_parser_audit_result.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

try:
    from scipy.stats import beta as _beta
except ImportError:
    _beta = None

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results"
SHEET = OUT / "v1_parser_audit_sheet.csv"
RESULT = OUT / "v1_parser_audit_result.json"
PILOT = ("qwen/qwen-2.5-7b-instruct", "2026-07-26")


def load_built_records():
    rows = []
    for p in sorted(RAW.glob("full_*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        if (s.get("model"), s.get("access_date")) == PILOT:
            continue
        model = s.get("model")
        for r in s.get("records", []):
            if r.get("status") != "built":
                continue
            if r.get("den_clean") is None or not r.get("trace"):
                continue
            rows.append(
                {
                    "model": model,
                    "record_id": r.get("record_id"),
                    "problem_id": r.get("problem_id"),
                    "seed": r.get("seed"),
                    "parser_den_clean": r.get("den_clean"),
                    "trace_text": r.get("trace", ""),
                }
            )
    return rows


def do_sample(n: int, seed: int) -> None:
    pool = load_built_records()
    if not pool:
        sys.exit(f"no built records under {RAW}")
    # Prefer one item per problem to reduce dependence; then fill to n.
    by_problem: dict[str, list] = {}
    for r in pool:
        by_problem.setdefault(r["problem_id"], []).append(r)
    rng = random.Random(seed)
    pids = sorted(by_problem)
    rng.shuffle(pids)
    picked = []
    for pid in pids:
        if len(picked) >= n:
            break
        picked.append(rng.choice(by_problem[pid]))
    if len(picked) < n:
        remaining = [r for r in pool if r not in picked]
        rng.shuffle(remaining)
        picked.extend(remaining[: n - len(picked)])
    out_rows = []
    for r in picked:
        out_rows.append(
            {
                **r,
                "human_denotation": "",
                "agree": "",
                "note": "",
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "record_id",
                "problem_id",
                "seed",
                "parser_den_clean",
                "trace_text",
                "human_denotation",
                "agree",
                "note",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)
    from collections import Counter

    print(f"wrote {SHEET}  ({len(out_rows)} rows)")
    print(Counter(r["model"] for r in out_rows))
    print("Fill human_denotation and agree (yes|no), then --analyze")


def _eq(a, b, tol=1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def cp_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    if _beta is None:
        p = k / n
        return (max(0.0, p - 1.96 * math.sqrt(p * (1 - p) / n)), min(1.0, p + 1.96 * math.sqrt(p * (1 - p) / n)))
    lo = float(_beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(_beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return (lo, hi)


def do_analyze() -> None:
    if not SHEET.exists():
        sys.exit(f"{SHEET} not found; run --sample first")
    rows = list(csv.DictReader(SHEET.open(encoding="utf-8")))
    filled = []
    for r in rows:
        agree = (r.get("agree") or "").strip().lower()
        human = (r.get("human_denotation") or "").strip()
        if not agree and not human:
            continue
        if agree not in {"yes", "no"}:
            # derive agree from human vs parser if human filled
            if human == "":
                continue
            agree = "yes" if _eq(human, r.get("parser_den_clean")) else "no"
        filled.append({**r, "agree": agree})
    if not filled:
        sys.exit(
            "no filled rows (need agree=yes|no and/or human_denotation). "
            "Nothing invented."
        )
    n = len(filled)
    k = sum(1 for r in filled if r["agree"] == "yes")
    lo, hi = cp_interval(k, n)
    by_model: dict[str, dict] = {}
    for r in filled:
        m = r["model"]
        slot = by_model.setdefault(m, {"n": 0, "agree": 0})
        slot["n"] += 1
        if r["agree"] == "yes":
            slot["agree"] += 1
    out = {
        "n_filled": n,
        "n_sheet": len(rows),
        "n_agree": k,
        "agreement_rate": k / n,
        "ci95": [lo, hi],
        "by_model": {
            m: {
                "n": v["n"],
                "agree": v["agree"],
                "rate": v["agree"] / v["n"] if v["n"] else None,
            }
            for m, v in by_model.items()
        },
    }
    RESULT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {RESULT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args()
    if a.sample:
        do_sample(a.n, a.seed)
    elif a.analyze:
        do_analyze()
    else:
        ap.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
