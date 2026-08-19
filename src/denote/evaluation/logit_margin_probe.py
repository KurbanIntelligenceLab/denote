#!/usr/bin/env python3
"""
denote_logit_margin_probe.py

Stricter OpenRouter feasibility test for AttriCoT / NLDD-style scoring than
denote_logprob_probe.py.

Asks, per model:
  1. Can we get token logprobs for an answer continuation under an *intact*
     CoT prefix and under a *corrupted* CoT prefix (same question, same
     intended answer string)?
  2. Can those form non-empty intact/corrupt margin vectors suitable for
     denote_baselines.attribution_curve / logit_margin_decay?
  3. Does the API expose anything resembling input attributions (CC-SHAP)?

This is NOT AttriCoT/NLDD/CC-SHAP. Do not fill H4 \\rc cells from this file
unless realistic_for_attricot_nldd is true and a separate collection is run.

Usage:
  python denote_logit_margin_probe.py
  python denote_logit_margin_probe.py --models meta-llama/llama-3.1-8b-instruct
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
OUT = ROOT / "results" / "logit_margin_probe.json"

DEFAULT_MODELS = (
    "meta-llama/llama-3.2-1b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "deepseek/deepseek-r1",
    "anthropic/claude-sonnet-5",
    "openai/gpt-5.6-luna",
)

QUESTION = "Compute 7 + 5."
INTACT_TRACE = (
    "sum = 7 + 5 = 12\n"
    "aside = 2 * 3 = 6\n"
    "The final answer is 12"
)
# Same surface length; wrong intermediate (corrupt computation).
CORRUPT_TRACE = (
    "sum = 7 + 5 = 19\n"
    "aside = 2 * 3 = 6\n"
    "The final answer is 12"
)
ANSWER_LINE = "The final answer is 12"
CONTINUE_SYSTEM = (
    "You are continuing from a supplied reasoning trace. "
    "Treat every equality in the trace as an asserted fact. "
    "Return exactly one line: The final answer is NUMBER"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token_logprobs(choice) -> list[dict]:
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    if not content:
        return []
    rows = []
    for tok in content:
        tops = getattr(tok, "top_logprobs", None) or []
        top_list = []
        for t in tops:
            top_list.append({
                "token": getattr(t, "token", None),
                "logprob": getattr(t, "logprob", None),
            })
        rows.append({
            "token": getattr(tok, "token", None),
            "logprob": getattr(tok, "logprob", None),
            "top_logprobs": top_list,
        })
    return rows


def _answer_token_margin(tokens: list[dict], answer_str: str = "12") -> list[float]:
    """Crude per-token margins at positions whose token appears in answer_str.

    Prefer logprob(correct-ish digit) - best competing top logprob when both
    exist; else best - second from top_logprobs. Empty if nothing usable.
    """
    margins: list[float] = []
    for tok in tokens:
        text = tok.get("token") or ""
        tops = tok.get("top_logprobs") or []
        vals = sorted(
            (float(t["logprob"]) for t in tops if t.get("logprob") is not None),
            reverse=True,
        )
        own = tok.get("logprob")
        # Prefer positions that look like answer digits
        if any(ch.isdigit() for ch in text) or text.strip() in answer_str:
            if own is not None and len(vals) >= 2:
                # margin vs runner-up (exclude own if it is best)
                best, second = vals[0], vals[1]
                margins.append(float(best - second))
            elif len(vals) >= 2:
                margins.append(float(vals[0] - vals[1]))
        elif len(vals) >= 2 and own is not None:
            margins.append(float(vals[0] - vals[1]))
    return margins


def _complete_with_logprobs(client: OpenAI, model: str, trace: str) -> dict:
    user = f"Question:\n{QUESTION}\n\nReasoning trace:\n{trace}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CONTINUE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=24,
            logprobs=True,
            top_logprobs=5,
            seed=0,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "text": None,
            "tokens": [],
            "margins": [],
        }
    choice = (resp.choices or [None])[0]
    if choice is None:
        return {"ok": False, "error": "no choices", "text": None, "tokens": [], "margins": []}
    text = (getattr(choice.message, "content", None) or "")[:200]
    tokens = _token_logprobs(choice)
    margins = _answer_token_margin(tokens)
    return {
        "ok": bool(tokens) and bool(margins),
        "error": None if tokens else "empty logprobs content",
        "text": text,
        "n_tokens": len(tokens),
        "margins": margins,
        "mean_margin": (sum(margins) / len(margins)) if margins else None,
    }


def _probe_cc_shap(client: OpenAI, model: str) -> dict:
    """Expect fail: routing APIs do not return input attributions."""
    out = {
        "attempted": True,
        "supported": False,
        "error": None,
        "note": (
            "CC-SHAP needs input attributions / explanation-prediction "
            "consistency. OpenRouter chat.completions has no standard "
            "attribution field; this probe only checks that."
        ),
    }
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi."}],
            temperature=0.0,
            max_tokens=8,
            # Non-standard knobs some local stacks use; should be ignored or error.
            extra_body={"attributions": True, "explain": True},
        )
        choice = (resp.choices or [None])[0]
        dumped = choice.model_dump() if choice is not None and hasattr(choice, "model_dump") else {}
        has_attr = any(
            k for k in dumped
            if "attr" in k.lower() or "shap" in k.lower() or "salienc" in k.lower()
        )
        # Also check response-level extras
        raw = resp.model_dump() if hasattr(resp, "model_dump") else {}
        has_attr = has_attr or any(
            k for k in raw
            if "attr" in str(k).lower() or "shap" in str(k).lower()
        )
        out["supported"] = bool(has_attr)
        if not has_attr:
            out["error"] = "no attribution/shap fields on chat.completions response"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def probe_model(client: OpenAI, model: str) -> dict:
    print(f"[logit-margin] {model} intact ...", flush=True)
    intact = _complete_with_logprobs(client, model, INTACT_TRACE)
    print(f"[logit-margin] {model} corrupt ...", flush=True)
    corrupt = _complete_with_logprobs(client, model, CORRUPT_TRACE)
    print(f"[logit-margin] {model} cc-shap ...", flush=True)
    cc = _probe_cc_shap(client, model)

    can_pair = bool(intact.get("margins")) and bool(corrupt.get("margins"))
    # Feedable into attribution_curve if we wrap as Rollout.margins = (intact, corrupt)
    realistic = can_pair and intact.get("ok") and corrupt.get("ok")

    note_parts = []
    if not intact.get("tokens") and not corrupt.get("tokens"):
        note_parts.append("API returned no token logprobs under either prefix")
    elif not can_pair:
        note_parts.append("could not form intact AND corrupt margin vectors")
    else:
        note_parts.append(
            "Got intact+corrupt generation-time top_logprobs margins on answer "
            "tokens — closer to NLDD inputs than the crude probe, but still NOT "
            "teacher-forced log P of fixed answer tokens / full AttriCoT"
        )
    if not cc.get("supported"):
        note_parts.append("CC-SHAP unsupported on this hop")

    return {
        "model": model,
        "intact": {k: v for k, v in intact.items() if k != "tokens"},
        "corrupt": {k: v for k, v in corrupt.items() if k != "tokens"},
        "intact_n_tokens": intact.get("n_tokens", 0),
        "corrupt_n_tokens": corrupt.get("n_tokens", 0),
        "can_form_intact_corrupt_margin_pair": can_pair,
        "realistic_for_attricot_nldd": realistic,
        "cc_shap": cc,
        "note": "; ".join(note_parts),
    }


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
        default_headers={"X-Title": "DENOTE-logit-margin-probe"},
    )

    rows = [probe_model(client, m) for m in models]
    usable = [r["model"] for r in rows if r["realistic_for_attricot_nldd"]]
    overall = bool(usable)

    result = {
        "generated_at": utc_now(),
        "purpose": (
            "OpenRouter intact-vs-corrupt answer-token logprob margin probe "
            "for AttriCoT/NLDD feasibility. Not a faithfulness measurement; "
            "do not fill H4 AttriCoT/NLDD/CC-SHAP from this alone."
        ),
        "question": QUESTION,
        "answer_line": ANSWER_LINE,
        "overall_realistic_for_attricot_nldd": overall,
        "models_realistic": usable,
        "models_not_realistic": [r["model"] for r in rows if not r["realistic_for_attricot_nldd"]],
        "cc_shap_overall": (
            "unsupported" if not any(r["cc_shap"].get("supported") for r in rows)
            else "unexpected_support"
        ),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(f"overall_realistic_for_attricot_nldd={overall}")
    print(f"models_realistic={usable or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
