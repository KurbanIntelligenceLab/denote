#!/usr/bin/env python3
"""Analyze tagged grammar raw JSON without touching Table 1.

Usage:
  python denote_grammar_analyze.py
  python denote_grammar_analyze.py --path results/raw/full_deepseek_deepseek-r1_grammar.json
  python denote_grammar_analyze.py --path a.json --path b.json
  python denote_grammar_analyze.py --all-grammar
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from denote_experiments import analyze_state, _clean_floats

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "results" / "raw"
DEFAULT = RAW_DIR / "full_deepseek_deepseek-r1_grammar.json"
OUT_SINGLE = ROOT / "results" / "grammar_summary.json"
OUT_PANEL = ROOT / "results" / "grammar_panel_summary.json"

KEYS = ("model", "n_raw", "n_gated", "gate_rate", "phi", "delta", "sigma")


def _enrich(row: dict, path: Path, state: dict) -> dict:
    row = dict(row)
    row["grammar_normalized"] = state.get("grammar_normalized")
    row["max_trace_tokens"] = state.get("max_trace_tokens")
    row["output_tag"] = state.get("output_tag") or "grammar"
    row["source_raw"] = str(path.as_posix())
    row["completed_at"] = state.get("completed_at")
    return row


def _analyze_path(path: Path) -> dict:
    state = json.loads(path.read_text(encoding="utf-8"))
    if not state.get("completed_at"):
        raise SystemExit(
            f"refusing: {path.name} has no completed_at "
            f"(n_records={len(state.get('records', []))})"
        )
    return _enrich(analyze_state(state), path, state)


def _discover_grammar_paths() -> list[Path]:
    paths = sorted(RAW_DIR.glob("full_*_grammar.json"))
    if not paths:
        raise SystemExit(f"no full_*_grammar.json under {RAW_DIR}")
    return paths


def _is_deepseek(path: Path, row: dict) -> bool:
    name = path.name.lower()
    model = str(row.get("model") or "").lower()
    return "deepseek" in name or "deepseek" in model


def main() -> int:
    p = argparse.ArgumentParser(
        description="Analyze grammar-tagged raw JSON; never touches Table 1 files."
    )
    p.add_argument(
        "--path",
        type=Path,
        action="append",
        dest="paths",
        help="Grammar raw JSON (repeatable). Default: DeepSeek grammar raw.",
    )
    p.add_argument(
        "--all-grammar",
        action="store_true",
        help="Analyze every results/raw/full_*_grammar.json",
    )
    p.add_argument(
        "--panel-out",
        type=Path,
        default=OUT_PANEL,
        help=f"Multi-model panel summary (default: {OUT_PANEL})",
    )
    p.add_argument(
        "--single-out",
        type=Path,
        default=OUT_SINGLE,
        help=f"DeepSeek back-compat summary (default: {OUT_SINGLE})",
    )
    args = p.parse_args()

    if args.all_grammar:
        paths = _discover_grammar_paths()
    elif args.paths:
        paths = [Path(x) for x in args.paths]
    else:
        paths = [DEFAULT]

    rows: list[dict] = []
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"missing: {path}")
        row = _analyze_path(path)
        rows.append(row)
        print({k: row.get(k) for k in KEYS})

    # Back-compat: always refresh grammar_summary.json when DeepSeek is present.
    deepseek_rows = [
        (path, row) for path, row in zip(paths, rows) if _is_deepseek(path, row)
    ]
    if len(paths) == 1 and deepseek_rows:
        args.single_out.write_text(
            json.dumps(_clean_floats(rows[0]), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote {args.single_out}")
    elif deepseek_rows:
        # Prefer the dedicated DeepSeek grammar raw if several DeepSeek paths.
        ds_path, ds_row = deepseek_rows[0]
        for path, row in deepseek_rows:
            if "deepseek-r1_grammar" in path.name:
                ds_path, ds_row = path, row
                break
        args.single_out.write_text(
            json.dumps(_clean_floats(ds_row), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote {args.single_out} (DeepSeek back-compat from {ds_path.name})")

    # Panel summary whenever more than one model, or --all-grammar.
    if len(rows) > 1 or args.all_grammar:
        panel = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_models": len(rows),
            "models": [_clean_floats(r) for r in rows],
            "index": {
                r["model"]: {
                    k: r.get(k)
                    for k in (
                        "n_raw",
                        "n_gated",
                        "gate_rate",
                        "phi",
                        "delta",
                        "sigma",
                        "phi_ci",
                        "delta_ci",
                        "sigma_ci",
                        "source_raw",
                    )
                }
                for r in rows
            },
        }
        args.panel_out.parent.mkdir(parents=True, exist_ok=True)
        args.panel_out.write_text(
            json.dumps(_clean_floats(panel), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote {args.panel_out}")
    elif len(rows) == 1 and not deepseek_rows:
        # Single non-DeepSeek path: still emit a one-row panel for convenience.
        panel = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_models": 1,
            "models": [_clean_floats(rows[0])],
            "index": {rows[0]["model"]: {k: rows[0].get(k) for k in KEYS}},
        }
        args.panel_out.parent.mkdir(parents=True, exist_ok=True)
        args.panel_out.write_text(
            json.dumps(_clean_floats(panel), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote {args.panel_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
