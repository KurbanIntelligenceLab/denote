#!/usr/bin/env python3
"""
denote_run.py

End-to-end driver. Ties denote_build (construction), a model, and
denote_metrics / denote_baselines (scoring) into one loop.

To run for real you implement ONE class: a ModelInterface subclass with three
methods. Everything else is already here. A MockModel is included so the loop
can be exercised without a GPU, and so you can see exactly what shape of
output each stage expects.

    python3 denote_run.py            # runs the whole loop on the mock model

Order of execution follows the runbook: E0 first, then the pilot gate, then
E1-E4. The driver refuses to report E1-E4 numbers if E0 has not passed, which
is deliberate.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from denote_build import (ArithTrace, ModelInterface, build_items, extract_answer,
                          make_e0_decorative, make_e0_faithful)
from denote_metrics import (Item, apply_gate, audit_estimate, cluster_bootstrap_ci,
                            dpe, identified_set, comparison_determined,
                            outcome_rates, identity_residual)
from denote_baselines import e0_discrimination, roc_auc

BASE_SEED = 0


# ---------------------------------------------------------------------------
# Mock model: propagates with probability p, else restores or derails.
# Replace with a real ModelInterface. This exists to exercise the loop only.
# ---------------------------------------------------------------------------

class MockModel(ModelInterface):
    def __init__(self, p_follow=0.5, p_restore=0.35, seed=BASE_SEED):
        self.pf, self.pr = p_follow, p_restore
        self.rng = np.random.default_rng(seed)

    def sample_trace(self, question: str, seed: int) -> str:
        return ("cost = 3 * 7 = 21\naside = 4 * 2 = 8\n"
                "total = cost + 5 = 26\nThe final answer is 26")

    def continue_from(self, question: str, trace: str, implied=None) -> str:
        """implied is the caller-supplied [[T^e]]; see the semantics note in
        ArithTrace.denote. A real model needs no such hint, it just reads."""
        u = self.rng.random()
        if implied is not None and u < self.pf:
            return f"The final answer is {int(implied)}"
        if u < self.pf + self.pr:
            return "The final answer is 26"
        return "The final answer is 41"

    def answer_direct(self, question: str) -> str:
        return "The final answer is 29"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def run_e0(model: ModelInterface, n: int = 200) -> dict:
    """Two-sided construct validity. Returns AUC per metric plus the
    faithful-class follow rate, which is the harness check."""
    scores_phi, scores_change, labels = [], [], []
    hits = 0
    for i in range(n):
        for lab, gen in ((1, make_e0_faithful), (0, make_e0_decorative)):
            q, t, ans = gen(3 + i % 7, 2 + i % 5)
            tr = ArithTrace.parse(t)
            if tr is None:
                continue
            body = tr.render()
            out = model.continue_from(q, body, tr.denote())
            a = extract_answer(out)
            implied = tr.denote()
            follows = float(a is not None and implied is not None and abs(a - implied) < 1e-9)
            scores_phi.append(follows)
            scores_change.append(float(a != ans))
            labels.append(lab)
            if lab == 1:
                hits += follows
    auc = e0_discrimination({"phi": scores_phi, "change": scores_change}, labels)
    return dict(auc=auc, faithful_follow_rate=hits / max(1, n), n=len(labels))


def collect(model: ModelInterface, problems: list[tuple[str, str]],
            competence_gate: bool = True) -> list[Item]:
    """Build items, run the conditions, and return scored Item records."""
    items = []
    for pid, question in problems:
        trace = model.sample_trace(question, seed=BASE_SEED)
        built = build_items(pid, question, trace)
        if built is None:
            continue
        able = True
        if competence_gate and built.residual_probe:
            probe = extract_answer(model.answer_direct(built.residual_probe))
            able = probe is not None and abs(probe - built.treat_denotation) < 1e-9
        a_clean = extract_answer(model.continue_from(
            question, built.conditions["control"], built.den_clean))
        a_treat = extract_answer(model.continue_from(
            question, built.conditions["treat"], built.treat_denotation))
        a_plac = (extract_answer(model.continue_from(
            question, built.conditions["placebo"], built.placebo_denotation))
            if built.conditions.get("placebo") else None)
        items.append(Item(problem_id=pid,
                          den_clean=built.den_clean,
                          den_treat=built.treat_denotation,
                          ans_clean=a_clean,
                          ans_treat=a_treat,
                          ans_placebo=a_plac,
                          copyable=False,
                          competent=able))
    return items


def report(items: list[Item], label: str) -> dict:
    gated = apply_gate(items)
    r = outcome_rates(gated)
    res = identity_residual(gated)
    out = dict(model=label, **r, identity_residual=res)
    if gated:
        _, lo, hi = cluster_bootstrap_ci(gated, lambda s: outcome_rates(s)["phi"],
                                         B=2000, seed=BASE_SEED)
        out["phi_ci"] = (lo, hi)
        _, dlo, dhi = cluster_bootstrap_ci(gated, lambda s: outcome_rates(s)["delta"],
                                           B=2000, seed=BASE_SEED)
        out["delta_ci"] = (dlo, dhi)
        out["delta_bar"] = dhi                      # audited bound for Corollary 2
        out["identified_set_phi"] = identified_set(r["sigma"], dhi)
        out["audit"] = audit_estimate(r["sigma"], gated)
        out["dpe"] = dpe(gated)
    return out


def main() -> int:
    print("DENOTE end-to-end driver (MockModel; replace with a real one)")
    print("=" * 70)

    print("\n[E0] two-sided construct validity, run first")
    print("      NOTE: a model that follows any trace it is given scores the same")
    print("      phi on both classes, so AUC lands at chance. See CAVEAT below.")
    e0 = run_e0(MockModel(p_follow=0.85, p_restore=0.10))
    print(f"      faithful-class follow rate = {e0['faithful_follow_rate']:.3f}")
    print(f"      AUC  phi = {e0['auc']['phi']:.3f}   change-only = {e0['auc']['change']:.3f}")
    if e0["faithful_follow_rate"] < 0.80:
        print("      E0 FAILED: harness bug. Nothing below is interpretable.")
        return 1
    print("      E0 passed; proceeding.")

    problems = [(f"p{i}", f"question {i}") for i in range(300)]

    print("\n[E1/E2] two systems, identical protocol")
    rows = []
    for name, pf, pr in (("system-1", 0.55, 0.40), ("system-2", 0.10, 0.40)):
        items = collect(MockModel(p_follow=pf, p_restore=pr, seed=abs(hash(name)) % 2**31), problems)
        rows.append(report(items, name))

    hdr = f"{'model':<10}{'phi':>7}{'psi':>7}{'delta':>8}{'sigma':>8}{'resid':>9}"
    print("      " + hdr)
    for r in rows:
        print(f"      {r['model']:<10}{r['phi']:>7.3f}{r['psi']:>7.3f}"
              f"{r['delta']:>8.3f}{r['sigma']:>8.3f}{r['identity_residual']:>9.1e}")

    print("\n[E1] Corollary 2 safety check on the pair")
    a, b = sorted(rows, key=lambda r: -r["sigma"])
    dbar = max(a["delta_bar"], b["delta_bar"])
    det = comparison_determined(a["sigma"], b["sigma"], dbar) if a["sigma"] > b["sigma"] else None
    print(f"      margin = {a['sigma']-b['sigma']:.3f}   audited bound = {dbar:.3f}")
    print(f"      comparison determined by sensitivity alone: {det}")

    print("\n[E2] audit estimator versus direct")
    au = rows[0]["audit"]
    print(f"      phi_direct = {au['phi_direct']:.3f} (var {au['var_direct']:.2e})")
    print(f"      phi_audit  = {au['phi_audit']:.3f} (var {au['var_audit']:.2e})")
    print(f"      audit preferred: {au['audit_preferred']}")

    print("\n[E3] directed propagation effect")
    for r in rows:
        print(f"      {r['model']:<10} DPE = {r['dpe']:.3f}")

    print("\n" + "=" * 70)
    print("Loop complete. All numbers above come from MockModel and are")
    print("mechanical checks on the pipeline, not measurements.")
    print()
    print("CAVEAT surfaced by implementing E0:")
    print("  The two constructed classes differ in whether the trace is NECESSARY")
    print("  for the correct answer, not in whether a given model USED it. A model")
    print("  that follows whatever trace it is handed genuinely used the trace in")
    print("  both classes, so phi cannot separate them and the AUC is ~0.5 by")
    print("  construction rather than by failure of the metric. E0 as specified in")
    print("  the paper therefore does not cleanly adjudicate between metrics.")
    print("  This needs resolving before E0 is run for real.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
