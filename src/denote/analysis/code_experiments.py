#!/usr/bin/env python3
"""Denote-Code experiment runner (second domain), mirroring
denote_experiments.py's architecture for the arithmetic panel.

Typical workflow
----------------
  python denote_code_experiments.py preflight
  python denote_code_experiments.py construction-survival
  python denote_code_experiments.py pilot --model qwen/qwen-2.5-7b-instruct --limit 30
  python denote_code_experiments.py full --limit 150 --seeds 0
  python denote_code_experiments.py analyze
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

from denote_code_build import (
    CodeCall,
    CodeItem,
    build_code_item,
    extract_code_answer,
    load_mbpp_problems,
)
from denote_experiments import (
    DEFAULT_MODELS,
    DEFAULT_PILOT_MODEL,
    write_json,
    _clean_floats,
    _slug,
    utc_now,
    ACCESS_DATE,
)
from denote_metrics import Item, apply_gate, identity_residual, outcome_rates
from denote_model import MissingOpenRouterKey, OpenRouterModel

RESULTS = Path("results")
RAW = RESULTS / "raw"
BASE_SEED = 0

CODE_TRACE_SYSTEM = """You solve Python programming problems by writing ONE complete function.

STRICT FORMAT RULES:
1. Return ONLY the function source. No markdown fences, no prose, no tests.
2. The function must be named to match the problem (use the name implied by the prompt / examples).
3. Prefer clear intermediate assignments so a reader can resume mid-function.
4. Do not include assert statements or if __name__ == '__main__' blocks.
5. Use only the Python standard library.

Example:
def total_cost(price, qty):
    tax = price * 0.1
    subtotal = price * qty
    return subtotal + tax * qty"""

CODE_CONTINUE_SYSTEM = """You are continuing from a supplied intermediate program state.
Treat every stated variable value as an asserted fact, even if it looks
inconsistent with how it might normally be computed. Mentally execute the
remaining statements exactly as given and report the outcome.
Return exactly one line: FINAL RESULT: <value>  (a JSON literal: number, string, list, true/false/null)"""

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```", re.I)
_DEF = re.compile(r"(^|\n)(def\s+\w+\s*\([\s\S]*)", re.M)


def render_condition(state: dict, tail_source: str) -> str:
    """Human-readable state+tail prefix for forced continuation."""
    payload = json.dumps(state, ensure_ascii=False, default=str)
    return f"STATE: {payload}\nTAIL:\n{tail_source}"


def extract_function_source(text: str) -> str | None:
    """Pull a single top-level function def out of a model completion."""
    if not text or not text.strip():
        return None
    m = _CODE_FENCE.search(text)
    body = m.group(1).strip() if m else text.strip()
    m2 = _DEF.search(body)
    if m2:
        body = m2.group(2).strip()
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return None
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(defs) != 1:
        return None
    # Re-slice from original body so we keep the exact source text.
    lines = body.splitlines()
    start = defs[0].lineno - 1
    end = defs[0].end_lineno
    return "\n".join(lines[start:end]).rstrip() + "\n"


class OpenRouterCodeModel(OpenRouterModel):
    """OpenRouterModel with code-domain system prompts."""

    def sample_trace(self, question: str, seed: int) -> str:
        return self._complete(
            purpose="code_sample",
            system=CODE_TRACE_SYSTEM,
            user=f"Problem:\n{question}",
            temperature=self.trace_temperature,
            seed=seed,
            max_tokens=self.max_trace_tokens,
        )

    def continue_from(self, question: str, trace: str, implied=None) -> str:
        del implied
        return self._complete(
            purpose="code_continue",
            system=CODE_CONTINUE_SYSTEM,
            user=f"Problem:\n{question}\n\nIntermediate state and remaining code:\n{trace}",
            temperature=0.0,
            seed=0,
            max_tokens=max(120, self.max_answer_tokens),
        )


def preflight_code() -> dict:
    import os
    from dotenv import load_dotenv

    load_dotenv(override=True)
    status = {
        "timestamp": utc_now(),
        "openrouter_key_present": bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
        "note": "Presence check only; pilot/full make live calls.",
    }
    write_json(RESULTS / "code_preflight.json", status)
    return status


def run_construction_survival(limit: int | None = None) -> dict:
    problems = load_mbpp_problems(limit)
    n = len(problems)
    n_call_parsed = 0
    n_built = 0
    rows = []
    for p in problems:
        call: CodeCall | None = p["call"]
        if call is None:
            rows.append({"problem_id": p["problem_id"], "status": "no_parseable_test_call"})
            continue
        n_call_parsed += 1
        try:
            item = build_code_item(p["problem_id"], p["prompt"], p["code"], call)
        except Exception as exc:
            rows.append({"problem_id": p["problem_id"], "status": f"construction_error:{exc}"})
            continue
        if item is None:
            rows.append({"problem_id": p["problem_id"], "status": "gate_rejected"})
            continue
        n_built += 1
        rows.append({
            "problem_id": p["problem_id"],
            "status": "built",
            "func_name": item.func_name,
            "resume_lineno": item.resume_lineno,
            "edited_var": item.edited_var,
            "den_clean": item.den_clean,
            "den_treat": item.treat_denotation,
            "has_placebo": item.conditions.get("placebo") is not None,
        })
    result = {
        "generated_at": utc_now(),
        "dataset": "mbpp_sanitized_test",
        "n_problems": n,
        "n_test_call_parsed": n_call_parsed,
        "n_built_c1_c3": n_built,
        "test_call_parse_rate": (n_call_parsed / n) if n else None,
        "build_rate": (n_built / n) if n else None,
        "note": (
            "Construction-only survival on reference solutions. "
            "No model called. Not a phi/delta/sigma measurement."
        ),
        "rows": rows,
    }
    write_json(RESULTS / "code_construction_survival.json", _clean_floats(result))
    print(f"[construction-survival] n={n} test_call_parsed={n_call_parsed} "
          f"({n_call_parsed / n:.1%}) built={n_built} ({n_built / n:.1%})")
    return result


def _load_code_panel(limit: int) -> list[dict]:
    """MBPP problems with a parseable literal test call, capped at limit.

    ``mbpp:test:260`` is excluded: OpenRouter repeatedly hangs indefinitely on
    ``code_sample`` for this prompt across models (observed on Llama-8B and
    Llama-70B), and the HTTP timeout does not always fire on that hop.
    """
    problems = [
        p for p in load_mbpp_problems()
        if p["call"] is not None and p["problem_id"] != "mbpp:test:260"
    ]
    return problems[:limit]


def collect_code_interventional(
    model: OpenRouterCodeModel,
    problems: list[dict],
    *,
    seeds: list[int],
    output_path: Path,
) -> dict:
    """Sample model solutions, build edits, score control/treat/placebo/copyable."""
    state = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {
        "model": model.model,
        "domain": "code",
        "access_date": ACCESS_DATE,
        "started_at": utc_now(),
        "records": [],
    }
    done = {r["record_id"] for r in state["records"]}
    for index, problem in enumerate(problems, start=1):
        for seed in seeds:
            rejected_id = f"{problem['problem_id']}:{seed}:rejected"
            built_prefix = f"{problem['problem_id']}:{seed}:"
            if rejected_id in done or any(
                rid.startswith(built_prefix) and not rid.endswith(":rejected") for rid in done
            ):
                print(
                    f"[code-collect] {model.model}  {index}/{len(problems)}  "
                    f"{problem['problem_id']} seed={seed}  skip",
                    flush=True,
                )
                continue
            print(
                f"[code-collect] {model.model}  {index}/{len(problems)}  "
                f"{problem['problem_id']} seed={seed}",
                flush=True,
            )
            built: CodeItem | None = None
            source = ""
            raw_trace = ""
            sample_api_error: str | None = None
            for attempt in range(3):
                print(f"  sample_code attempt {attempt + 1}/3 ...", flush=True)
                try:
                    raw_trace = model.sample_trace(problem["prompt"], seed + 1000 * attempt)
                except Exception as exc:
                    # Tenacity already retried; skip pair instead of aborting the model.
                    sample_api_error = str(exc)
                    print(f"  sample API error after retries: {sample_api_error}", flush=True)
                    break
                source = extract_function_source(raw_trace) or ""
                if not source:
                    continue
                try:
                    built = build_code_item(
                        problem["problem_id"], problem["prompt"], source, problem["call"]
                    )
                except Exception as exc:
                    print(f"  build error: {exc}", flush=True)
                    built = None
                if built is not None:
                    print(
                        f"  built edit on {built.edited_var} @ line {built.resume_lineno}",
                        flush=True,
                    )
                    break
            if sample_api_error is not None:
                state["records"].append({
                    "record_id": rejected_id,
                    "problem_id": problem["problem_id"],
                    "dataset": "mbpp",
                    "seed": seed,
                    "trace": raw_trace,
                    "source": source,
                    "status": "api_error_skipped",
                    "parsed": bool(source),
                    "error": sample_api_error,
                })
                done.add(rejected_id)
                write_json(output_path, state)
                continue
            if built is None:
                state["records"].append({
                    "record_id": rejected_id,
                    "problem_id": problem["problem_id"],
                    "dataset": "mbpp",
                    "seed": seed,
                    "trace": raw_trace,
                    "source": source,
                    "status": "C1_or_trace_consistency_rejected",
                    "parsed": bool(source),
                })
                done.add(rejected_id)
                write_json(output_path, state)
                print("  rejected", flush=True)
                continue

            record_id = f"{problem['problem_id']}:{seed}:{built.resume_lineno}"
            if record_id in done:
                continue
            answers: dict[str, Any] = {}
            outputs: dict[str, Any] = {}
            rendered: dict[str, Any] = {}
            try:
                for condition in ("control", "treat", "placebo", "copyable"):
                    cond = built.conditions.get(condition)
                    if not cond:
                        answers[condition] = None
                        outputs[condition] = None
                        rendered[condition] = None
                        continue
                    state_dict, tail = cond
                    text = render_condition(state_dict, tail)
                    rendered[condition] = text
                    outputs[condition] = model.continue_from(problem["prompt"], text)
                    answers[condition] = extract_code_answer(outputs[condition])
                # C4: can the model evaluate the treated residual directly?
                probe = (
                    "Read this intermediate program state and remaining code, "
                    "then report the FINAL RESULT as a JSON literal.\n"
                    + (rendered.get("treat") or "")
                )
                competent_answer = extract_code_answer(model.answer_direct(probe))
                competent = competent_answer == built.treat_denotation
            except Exception as exc:
                # After tenacity retries (incl. OpenRouter no-choices), skip this
                # pair so the rest of the panel can finish; resume will not retry.
                err = str(exc)
                print(f"  API error after retries; skipping pair: {err}", flush=True)
                state["records"].append({
                    "record_id": rejected_id,
                    "problem_id": problem["problem_id"],
                    "dataset": "mbpp",
                    "seed": seed,
                    "trace": raw_trace,
                    "source": source,
                    "status": "api_error_skipped",
                    "parsed": bool(source),
                    "error": err,
                    "treatment_step_idx": built.resume_lineno,
                    "edited_var": built.edited_var,
                })
                done.add(rejected_id)
                write_json(output_path, state)
                continue
            state["records"].append({
                "record_id": record_id,
                "problem_id": problem["problem_id"],
                "dataset": "mbpp",
                "seed": seed,
                "question": problem["prompt"],
                "gold": problem["call"].expected,
                "trace": raw_trace,
                "source": source,
                "status": "built",
                "den_clean": built.den_clean,
                "den_treat": built.treat_denotation,
                "den_placebo": built.placebo_denotation,
                "den_copyable": built.copyable_denotation,
                "answers": answers,
                "outputs": outputs,
                "rendered": rendered,
                "competent": competent,
                "competent_answer": competent_answer,
                "treatment_step_idx": built.resume_lineno,
                "edited_var": built.edited_var,
                "trace_length": built.trace_length,
            })
            done.add(record_id)
            write_json(output_path, state)
            print(
                f"  saved {record_id}  clean={answers.get('control')} "
                f"treat={answers.get('treat')} competent={competent}",
                flush=True,
            )
    state["completed_at"] = utc_now()
    write_json(output_path, state)
    return state


def records_to_items(records: list[dict]) -> list[Item]:
    items = []
    for row in records:
        if row.get("status") != "built":
            continue
        answers = row.get("answers", {})
        items.append(Item(
            problem_id=row["problem_id"],
            den_clean=row.get("den_clean"),
            den_treat=row.get("den_treat"),
            ans_clean=answers.get("control"),
            ans_treat=answers.get("treat"),
            ans_placebo=answers.get("placebo"),
            copyable=False,
            competent=row.get("competent", False),
        ))
    return items


def analyze_code_state(state: dict) -> dict:
    raw_items = records_to_items(state["records"])
    gated = apply_gate(raw_items, require_c4=False)
    rates = outcome_rates(gated)
    residual = identity_residual(gated) if gated else 0.0
    if gated and residual >= 1e-12:
        raise RuntimeError(f"code T2: identity residual {residual} exceeds 1e-12")
    n_problems = len({r["problem_id"] for r in state["records"]})
    n_parsed = len({
        r["problem_id"] for r in state["records"]
        if r.get("status") == "built" or r.get("parsed")
    })
    n_built = len({r["problem_id"] for r in state["records"] if r.get("status") == "built"})
    return _clean_floats({
        "model": state["model"],
        "domain": "code",
        "access_date": state.get("access_date"),
        "n_raw": len(raw_items),
        "n_gated": len(gated),
        "n_problems": n_problems,
        "parse_rate_c1": n_parsed / n_problems if n_problems else 0.0,
        "build_rate": n_built / n_problems if n_problems else 0.0,
        "gate_rate": len(gated) / len(raw_items) if raw_items else 0.0,
        "identity_residual": residual,
        **{k: (int(v) if k == "n" else v) for k, v in rates.items()},
    })


def run_pilot(model_name: str, limit: int) -> dict:
    problems = _load_code_panel(limit)
    model = OpenRouterCodeModel(model_name)
    output = RAW / f"code_pilot_{_slug(model_name)}.json"
    state = collect_code_interventional(
        model, problems, seeds=[BASE_SEED], output_path=output
    )
    analysis = analyze_code_state(state)
    items = records_to_items(state["records"])
    gated = apply_gate(items, require_c4=False)
    residual = identity_residual(gated) if gated else 0.0
    result = _clean_floats({
        **analysis,
        "go": (
            analysis["parse_rate_c1"] >= 0.2
            and analysis["build_rate"] >= 0.1
            and residual < 1e-12
            and len(gated) > 0
        ),
        "go_note": (
            "Code pilot GO: parse>=0.2, build>=0.1, identity residual 0, "
            "at least one gated item. Thresholds are looser than math because "
            "MBPP reference construction itself is ~0.25."
        ),
    })
    write_json(RESULTS / "code_pilot_summary.json", result)
    return result


def run_full(models: list[str], limit: int, seeds: list[int]) -> dict:
    pilot_path = RESULTS / "code_pilot_summary.json"
    if not pilot_path.exists() or not json.loads(pilot_path.read_text(encoding="utf-8")).get("go"):
        raise RuntimeError("Code pilot GO has not passed; refusing full panel.")
    problems = _load_code_panel(limit)
    write_json(RESULTS / "code_problem_panel.json", {
        "n": len(problems),
        "ids": [p["problem_id"] for p in problems],
    })
    failures = []
    summaries = []
    for model_name in models:
        output = RAW / f"code_full_{_slug(model_name)}.json"
        print(f"[code-full] starting {model_name}", flush=True)
        try:
            model = OpenRouterCodeModel(model_name)
            state = collect_code_interventional(
                model, problems, seeds=seeds, output_path=output
            )
            summaries.append(analyze_code_state(state))
            print(f"[code-full] finished {model_name}", flush=True)
        except Exception as exc:
            failures.append({"model": model_name, "error": str(exc)})
            print(f"[code-full] FAILED {model_name}: {exc}", flush=True)
    summary = {
        "generated_at": utc_now(),
        "domain": "code",
        "models": summaries,
        "failures": failures,
    }
    write_json(RESULTS / "code_summary.json", _clean_floats(summary))
    return summary


def analyze_all() -> dict:
    summaries = []
    for path in sorted(RAW.glob("code_full_*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        if "records" in state:
            summaries.append(analyze_code_state(state))
    summary = {
        "generated_at": utc_now(),
        "domain": "code",
        "models": summaries,
    }
    write_json(RESULTS / "code_summary.json", _clean_floats(summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DENOTE-CODE runner (second domain).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="Report API-key presence")
    surv = sub.add_parser("construction-survival", help="Model-free MBPP construction yield")
    surv.add_argument("--limit", type=int, default=None)
    pilot = sub.add_parser("pilot", help="GO/NO-GO on a small MBPP panel")
    pilot.add_argument("--model", default=DEFAULT_PILOT_MODEL)
    pilot.add_argument("--limit", type=int, default=30)
    full = sub.add_parser("full", help="Full multi-model code panel (requires pilot GO)")
    full.add_argument("--models", default=",".join(DEFAULT_MODELS))
    full.add_argument("--limit", type=int, default=150)
    full.add_argument("--seeds", default="0")
    sub.add_parser("analyze", help="Score all results/raw/code_full_*.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "preflight":
            result = preflight_code()
        elif args.command == "construction-survival":
            result = run_construction_survival(args.limit)
        elif args.command == "pilot":
            result = run_pilot(args.model, args.limit)
        elif args.command == "full":
            models = [m.strip() for m in args.models.split(",") if m.strip()]
            seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
            result = run_full(models, args.limit, seeds)
        else:
            result = analyze_all()
    except MissingOpenRouterKey as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    summary = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(_clean_floats(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
