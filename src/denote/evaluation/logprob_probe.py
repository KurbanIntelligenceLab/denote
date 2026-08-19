#!/usr/bin/env python3
"""
denote_logprob_probe.py

Side-by-side probe: can OpenRouter return logprobs for panel models, enough
to even *attempt* AttriCoT/NLDD-style margin scoring without a local GPU?

This is NOT AttriCoT/NLDD. Those need complete (or at least answer-token)
logit margins under intact vs corrupted prefixes. This script only checks:
  1. Does OpenRouter accept logprobs=true / top_logprobs for our models?
  2. Do we get non-null token logprobs back?
  3. Can we form a crude per-token margin from top_logprobs (best - second)?

Writes results/logprob_probe.json. Safe to run while other collection jobs
are live (tiny request volume: one short completion per model).

Usage:
  python denote_logprob_probe.py
  python denote_logprob_probe.py --models qwen/qwen-2.5-7b-instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "logprob_probe.json"

DEFAULT_MODELS = (
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "openai/gpt-5.6-luna",
    "anthropic/claude-sonnet-5",
    "deepseek/deepseek-r1",
)

PROMPT = (
    "Solve: 3 + 4. Return exactly one line: The final answer is NUMBER"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def crude_margins(logprobs_content) -> list[float]:
    """Best-minus-second from top_logprobs, per generated token. Returns []."""
    if not logprobs_content:
        return []
    out = []
    for tok in logprobs_content:
        tops = getattr(tok, "top_logprobs", None) or []
        if len(tops) >= 2:
            vals = sorted((getattr(t, "logprob", None) for t in tops), reverse=True)
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                out.append(float(vals[0] - vals[1]))
        elif getattr(tok, "logprob", None) is not None:
            # single returned logprob — no margin, record as None-skip
            pass
    return out


def probe_model(client: OpenAI, model: str) -> dict:
    row = {
        "model": model,
        "ok": False,
        "error": None,
        "has_logprobs_field": False,
        "n_tokens_with_logprob": 0,
        "n_tokens_with_top": 0,
        "crude_margins": [],
        "mean_crude_margin": None,
        "sample_text": None,
        "note": None,
    }
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return exactly one line."},
                {"role": "user", "content": PROMPT},
            ],
            temperature=0.0,
            max_tokens=32,
            logprobs=True,
            top_logprobs=5,
            seed=1,
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["note"] = "request rejected or failed; model/provider may not support logprobs"
        return row

    choice = (resp.choices or [None])[0]
    if choice is None:
        row["error"] = "no choices"
        return row
    msg = choice.message
    row["sample_text"] = (getattr(msg, "content", None) or "")[:200]
    lp = getattr(choice, "logprobs", None)
    row["has_logprobs_field"] = lp is not None
    content = getattr(lp, "content", None) if lp is not None else None
    if not content:
        row["note"] = (
            "API accepted logprobs=true but returned empty/null content — "
            "not usable for margins via this hop"
        )
        return row
    row["n_tokens_with_logprob"] = sum(
        1 for t in content if getattr(t, "logprob", None) is not None
    )
    row["n_tokens_with_top"] = sum(
        1 for t in content if getattr(t, "top_logprobs", None)
    )
    margins = crude_margins(content)
    row["crude_margins"] = margins
    if margins:
        row["mean_crude_margin"] = sum(margins) / len(margins)
        row["ok"] = True
        row["note"] = (
            "Got top_logprobs; crude best-vs-second margins are available. "
            "This is NOT yet AttriCoT/NLDD (those need intact-vs-corrupted "
            "answer margins on the same items), but it unblocks an API pilot."
        )
    else:
        row["note"] = "logprobs present but could not form top-2 margins"
        row["ok"] = row["n_tokens_with_logprob"] > 0
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    load_dotenv(ROOT / ".env", override=True)
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("BLOCKED: no OPENROUTER_API_KEY", file=sys.stderr)
        return 2
    client = OpenAI(
        api_key=key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        timeout=60.0,
        default_headers={"X-Title": "DENOTE-logprob-probe"},
    )

    rows = []
    for model in models:
        print(f"[probe] {model} ...", flush=True)
        row = probe_model(client, model)
        rows.append(row)
        status = "OK" if row["ok"] else "NO"
        print(
            f"  {status}  top={row['n_tokens_with_top']}  "
            f"mean_margin={row['mean_crude_margin']}  "
            f"err={row['error']}",
            flush=True,
        )

    usable = [r["model"] for r in rows if r["ok"]]
    result = {
        "generated_at": utc_now(),
        "purpose": (
            "Side-by-side OpenRouter logprobs probe for AttriCoT/NLDD feasibility "
            "without local GPU. Not a faithfulness measurement."
        ),
        "models_usable_for_api_margin_pilot": usable,
        "models_not_usable": [r["model"] for r in rows if not r["ok"]],
        "cc_shap_note": (
            "CC-SHAP needs input attributions, not output logprobs; "
            "this probe does not help CC-SHAP."
        ),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(f"usable for API margin pilot: {usable or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
