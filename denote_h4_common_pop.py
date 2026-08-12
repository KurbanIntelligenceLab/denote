#!/usr/bin/env python3
"""
denote_h4_common_pop.py

H4 ("does the follow rate discriminate at least as well as existing
metrics?") on the one population that is actually API-feasible right now:
the eight models already scored in ``results/summary.json``, where phi and
the Lanham/hint baseline panel were all collected on the same problems.

Why this is a different table from the one drafted in denote_main.tex
------------------------------------------------------------------------
The worksheet table asks for an AUC: each metric's ability to separate two
*labelled* construct-validity classes (E0's faithful-by-construction vs
decorative-by-construction items). That requires running every baseline on
the SAME synthetic E0 items scored per model in ``e0_harness()`` -- which
needs live model calls this environment cannot make (no
OPENROUTER_API_KEY). Faking labels or reusing E0 numbers from a different
population would misrepresent H4 as tested when it is not.

What IS answerable without a new model call: analyze_baselines() already
scored early answering / adding mistakes / paraphrasing / filler tokens /
hint verbalization for every model in the SAME run that produced phi
(``results/summary.json``). ``denote_baselines.metric_agreement`` was
written to compare phi's *cross-model ranking* against any other metric's
cross-model ranking (it already does this for sigma inside
``denote_experiments.analyze_all``); this script is the one line of wiring
that runs it against the baseline panel too, which was the actual gap
(``per_model`` inside ``analyze_all`` only ever populated ``phi``/``sigma``).

This is a real, non-invented number: Spearman rank agreement between phi and
each baseline across the models with both defined, plus which model pairs
the two metrics order oppositely. It is NOT an AUC and does not by itself
settle H4 in the sense the worksheet poses it; that half (AttriCoT, NLDD,
CC-SHAP, and the labelled-E0 AUC comparison) stays explicitly open and is
reported as such, not silently dropped.

Run:
    python3 denote_h4_common_pop.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from denote_baselines import metric_agreement, spearman
from denote_experiments import SUMMARY, write_json, utc_now, _clean_floats

OUT = Path("results/h4_common_pop.json")

BASELINE_FIELDS = ("adding_mistakes", "paraphrasing", "filler_tokens", "hint_verbalization")


def _extract_per_model(summary: dict) -> dict[str, dict[str, float]]:
    """One row per model: phi, sigma, and every scalar baseline already
    collected in analyze_baselines(). early_answering is nested (curve+auc);
    the AUC field is what is comparable to the others as a single number.

    When: build_h4_common_pop(), once per run.
    """
    def _num(value):
        return float(value) if isinstance(value, (int, float)) and value is not None else float("nan")

    out = {}
    for row in summary.get("models", []):
        baselines = row.get("baselines") or {}
        entry = {"phi": _num(row.get("phi")), "sigma": _num(row.get("sigma"))}
        ea = baselines.get("early_answering")
        entry["early_answering_auc"] = _num(ea.get("auc")) if isinstance(ea, dict) else float("nan")
        for field in BASELINE_FIELDS:
            entry[field] = _num(baselines.get(field))
        out[row["model"]] = entry
    return out


def build_h4_common_pop() -> dict:
    """Wire phi vs every already-collected baseline on the shared model
    population; write results/h4_common_pop.json.

    When: python3 denote_h4_common_pop.py, after `analyze` has produced
    results/summary.json (both already true in this checkout).
    """
    if not SUMMARY.exists():
        raise FileNotFoundError(
            f"{SUMMARY} not found; run `python denote_experiments.py analyze` first."
        )
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    per_model = _extract_per_model(summary)
    agreement = metric_agreement(per_model)

    metric_names = [m for m in per_model[next(iter(per_model))] if m != "phi"]
    phi_by_model = {m: v["phi"] for m, v in per_model.items()}
    rows = []
    for name in metric_names:
        vals = {m: v.get(name) for m, v in per_model.items()}
        common = [
            m for m in per_model
            if phi_by_model[m] is not None and np.isfinite(phi_by_model[m])
            and vals[m] is not None and np.isfinite(vals[m])
        ]
        rho = agreement.get("spearman_vs_phi", {}).get(name)
        pairs = agreement.get("disagreeing_pairs", {}).get(name, [])
        rows.append({
            "metric": name,
            "spearman_vs_phi": rho,
            "common_population_n_models": len(common),
            "n_disagreeing_pairs": len(pairs),
            "n_possible_pairs": len(common) * (len(common) - 1) // 2,
            "disagreeing_pairs": pairs,
            "per_model_value": vals,
        })

    result = {
        "generated_at": utc_now(),
        "unit_of_analysis": (
            "models (n=8 collected; rows report how many have both phi and "
            "the given baseline finite), not items -- see module docstring "
            "for why this is not the labelled-E0 AUC the worksheet table asks for."
        ),
        "phi_by_model": phi_by_model,
        "metrics": rows,
        "unavailable": {
            "AttriCoT": "OpenRouter does not expose the required complete logit margins.",
            "NLDD": "OpenRouter does not expose the required complete logit margins.",
            "CC-SHAP": "OpenRouter does not expose input attributions.",
            "labelled_E0_AUC_vs_baselines": (
                "e0_harness() scores only phi/change-from-gold on synthetic "
                "items per model; the baseline panel was never re-run on "
                "those same synthetic items, which would need new live model "
                "calls (blocked here: no OPENROUTER_API_KEY)."
            ),
        },
    }
    write_json(OUT, _clean_floats(result))
    return result


def _print_table(result: dict) -> None:
    print(f"{'metric':<22}{'spearman vs phi':>16}{'n models':>10}{'disagree/possible':>20}")
    for row in result["metrics"]:
        rho = row["spearman_vs_phi"]
        rho_s = "nan" if rho is None or not np.isfinite(rho) else f"{rho:.3f}"
        print(f"{row['metric']:<22}{rho_s:>16}{row['common_population_n_models']:>10}"
             f"{row['n_disagreeing_pairs']:>10}/{row['n_possible_pairs']:<8}")


def main() -> int:
    result = build_h4_common_pop()
    _print_table(result)
    print(f"\nwrote {OUT}")
    print("NOTE: this is cross-model rank agreement, not a labelled-E0 AUC. "
         "AttriCoT/NLDD/CC-SHAP and the AUC comparison remain open (see "
         "result['unavailable']).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
