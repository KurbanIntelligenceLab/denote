#!/usr/bin/env python3
"""One-off: compute the legacy head-to-head numbers now that the legacy arm
is scored (denote_legacy_score.py). Mirrors denote_tdsc_verify.py's
gate_and_score() gating (C1: status == 'built'; C2: control answer agrees
with the trace's own denotation) so the population is exactly the one
Table I's sigma is computed from -- only the scored arm differs (legacy vs
treat). Standard library only, deliberately not importing the rest of the
bundle, matching denote_tdsc_verify.py's own "second opinion" convention.

Usage: python compute_legacy_summary.py
Writes: results/legacy_summary.json
"""
import json
import os

RAW_DIR = os.path.join("results", "raw")

SHORT = {
    "meta-llama/llama-3.1-70b-instruct": "Llama-70B",
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "meta-llama/llama-3.2-1b-instruct": "Llama-1B",
    "qwen/qwen-2.5-72b-instruct": "Qwen-72B",
    "qwen/qwen-2.5-7b-instruct": "Qwen-7B",
    "deepseek/deepseek-r1": "DeepSeek-R1",
    "anthropic/claude-sonnet-5": "Claude-5",
    "openai/gpt-5.6-luna": "GPT-5.6",
}
PILOT = ("qwen/qwen-2.5-7b-instruct", "2026-07-26")


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def gate_and_score_legacy(records, policy="worst"):
    """Same C1-C2 gate as gate_and_score() in denote_tdsc_verify.py, scored
    against answers['legacy'] instead of answers['treat']."""
    gated = []
    for r in records:
        if r.get("status") != "built":
            continue
        ans = r.get("answers") or {}
        den_clean = r.get("den_clean")
        if ans.get("control") is None or den_clean is None:
            continue
        if not close(ans["control"], den_clean):
            continue
        gated.append(r)

    follow = restore = derail = no_response = 0
    problems = set()
    for r in gated:
        al = (r.get("answers") or {}).get("legacy")
        if al is None:
            no_response += 1
            if policy == "complete":
                continue
            derail += 1
            problems.add(r.get("problem_id"))
            continue
        problems.add(r.get("problem_id"))
        if close(al, r["den_treat"]):
            follow += 1
        elif close(al, r["den_clean"]):
            restore += 1
        else:
            derail += 1

    n = follow + restore + derail
    if n == 0:
        return dict(n=0, n_gated=len(gated), sigma=None, phi=None, delta=None,
                    psi=None, follow=0, restore=0, derail=0,
                    no_response=no_response, nprob=0)
    return dict(n=n, n_gated=len(gated), sigma=(follow + derail) / n,
                phi=follow / n, delta=derail / n, psi=restore / n,
                follow=follow, restore=restore, derail=derail,
                no_response=no_response, nprob=len(problems))


def main():
    out = {}
    for fn in sorted(os.listdir(RAW_DIR)):
        if not (fn.startswith("full_") and fn.endswith(".json")):
            continue
        if "_grammar" in fn:
            continue
        path = os.path.join(RAW_DIR, fn)
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        if (blob.get("model"), blob.get("access_date")) == PILOT:
            continue
        name = SHORT.get(blob["model"], blob["model"])
        scored = gate_and_score_legacy(blob["records"])
        scored["file"] = fn
        scored["model"] = blob["model"]
        out[name] = scored

    result = {
        "generated_by": "compute_legacy_summary.py",
        "generated_note": (
            "Legacy head-to-head numbers computed after denote_legacy_score.py "
            "scored the legacy arm on the 8 canonical models. Gating (C1-C2) "
            "mirrors denote_tdsc_verify.py's gate_and_score() exactly; only the "
            "scored arm differs (legacy vs treat), so sigma_legacy is a "
            "like-for-like comparison against Table I's sigma."
        ),
        "models": out,
    }
    with open(os.path.join("results", "legacy_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"{'model':12s} {'n':>5s} {'n_gated':>8s} {'sigma':>7s} {'phi':>7s} {'delta':>7s} {'psi':>7s}")
    for name, s in out.items():
        if s["n"] == 0:
            print(f"{name:12s} {'--':>5s} {s['n_gated']:>8d} {'--':>7s} {'--':>7s} {'--':>7s} {'--':>7s}")
        else:
            print(f"{name:12s} {s['n']:>5d} {s['n_gated']:>8d} {s['sigma']:>7.3f} {s['phi']:>7.3f} {s['delta']:>7.3f} {s['psi']:>7.3f}")
    print("\nWrote results/legacy_summary.json")


if __name__ == "__main__":
    main()
