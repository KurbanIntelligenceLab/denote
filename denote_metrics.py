#!/usr/bin/env python3
"""
denote_metrics.py

Reference implementation of the quantities defined in
"Corruption Scores Are Upper Bounds: Sharp Identification and Cheap
Audits for Chain-of-Thought Faithfulness".

Implements
----------
  outcome classification            follow / restore / derail
  outcome rates                     phi, psi, delta
  legacy sensitivity                sigma
  identity check                    sigma == phi + delta   (Proposition 1)
  copy-explicability predicate      C3
  directed propagation effect       DPE  (Definition 1)
  on-policy estimator               DPE_obs
  clustered bootstrap CIs           resampling at the source-problem level
  paired sign-flip permutation test cluster-aware
  Holm step-down correction         across models

Dependencies: numpy only (plus the standard library).

IMPORTANT
---------
This module COMPUTES quantities from data you supply. It contains no results
from the paper and no measured numbers. The smoke test at the bottom runs on
synthetic data with planted, known ground truth; it verifies that the
estimators recover what was planted. It is a check on the code, not evidence
for any empirical claim in the paper.

Run the smoke test:
    python3 denote_metrics.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

BASE_SEED = 0

FOLLOW, RESTORE, DERAIL = "follow", "restore", "derail"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Item:
    """One benchmark item after gating.

    problem_id : source problem, used as the bootstrap cluster
    den_clean  : [[T]],   denotation of the unedited trace
    den_treat  : [[T^e]], denotation of the treatment-edited trace (v_e)
    ans_clean  : A,       model answer on the unedited trace
    ans_treat  : A^e,     model answer on the treatment-edited trace
    ans_placebo: A^p,     model answer on the surface-matched placebo trace
    copyable   : True iff v_e appears as a literal in T^e (violates C3)
    competent  : True iff the model passes the C4 competence probe
    """
    problem_id: str
    den_clean: object
    den_treat: object
    ans_clean: object
    ans_treat: object
    ans_placebo: object = None
    copyable: bool = False
    competent: bool = True


# ---------------------------------------------------------------------------
# Gating (inclusion conditions C1-C4)
# ---------------------------------------------------------------------------

def is_copy_explicable(target, literals: Iterable) -> bool:
    """C3 predicate. True when the treatment target is already written in the
    edited trace, so a copy heuristic could produce it without computing."""
    return any(_eq(target, lit) for lit in literals)


def passes_gate(it: Item, require_c3: bool = True, require_c4: bool = True) -> bool:
    """C1 is enforced upstream by the parser (unparseable traces never become
    Items). C2, C3, C4 are enforced here."""
    if it.den_clean is None or it.den_treat is None:
        return False                                    # C1
    if not _eq(it.ans_clean, it.den_clean):
        return False                                    # C2
    if _eq(it.den_treat, it.den_clean):
        return False                                    # not a treatment edit
    if require_c3 and it.copyable:
        return False                                    # C3
    if require_c4 and not it.competent:
        return False                                    # C4
    return True


def apply_gate(items: Sequence[Item], **kw) -> list[Item]:
    return [it for it in items if passes_gate(it, **kw)]


# ---------------------------------------------------------------------------
# Outcome classification and rates
# ---------------------------------------------------------------------------

def classify(it: Item, answer=None) -> str:
    """Assign one of the three mutually exclusive outcomes. Requires a
    treatment edit (den_treat != den_clean), which apply_gate enforces."""
    a = it.ans_treat if answer is None else answer
    if _eq(a, it.den_treat):
        return FOLLOW
    if _eq(a, it.den_clean):
        return RESTORE
    return DERAIL


def outcome_rates(items: Sequence[Item]) -> dict:
    """Return phi, psi, delta and the legacy sensitivity sigma.

    Under the gate, A = [[T]], so sigma = P(A^e != A) = phi + delta exactly.
    We compute sigma directly from answer inequality rather than from the
    identity, so that identity_residual() is a real check and not a tautology.
    """
    if not items:
        return dict(n=0, phi=np.nan, psi=np.nan, delta=np.nan, sigma=np.nan)
    labels = [classify(it) for it in items]
    n = len(items)
    phi = sum(l == FOLLOW for l in labels) / n
    psi = sum(l == RESTORE for l in labels) / n
    delta = sum(l == DERAIL for l in labels) / n
    sigma = sum(not _eq(it.ans_treat, it.ans_clean) for it in items) / n
    return dict(n=n, phi=phi, psi=psi, delta=delta, sigma=sigma)


def identity_residual(items: Sequence[Item]) -> float:
    """|sigma - (phi + delta)|. Proposition 1 says this is exactly zero on any
    gated sample. A nonzero value means the gate was not applied or C2 was
    violated, and is a bug signal."""
    r = outcome_rates(items)
    return abs(r["sigma"] - (r["phi"] + r["delta"]))


# ---------------------------------------------------------------------------
# Directed propagation effect
# ---------------------------------------------------------------------------

def dpe(items: Sequence[Item]) -> float:
    """DPE = mean_i [ 1{A^e_i = v_i} - 1{A^p_i = v_i} ].

    Both arms are scored against the same target v_i = den_treat, so the
    placebo term is the rate of emitting the treatment target when the trace
    does not compute it."""
    paired = [it for it in items if it.ans_placebo is not None]
    if not paired:
        return float("nan")
    d = [float(_eq(it.ans_treat, it.den_treat)) - float(_eq(it.ans_placebo, it.den_treat))
         for it in paired]
    return float(np.mean(d))


def dpe_per_item(items: Sequence[Item]) -> np.ndarray:
    paired = [it for it in items if it.ans_placebo is not None]
    return np.array([float(_eq(it.ans_treat, it.den_treat))
                     - float(_eq(it.ans_placebo, it.den_treat)) for it in paired])


def observational_dpe(traces: Sequence[tuple]) -> float:
    """On-policy estimator. Input is a sequence of (problem_id, denotation,
    answer) triples sampled from the model with no intervention.

    For each problem and each candidate value v attained by at least one
    sampled trace:
        P(A = v | [[T]] = v) - P(A = v | [[T]] != v)
    averaged over (problem, v) cells that have both arms populated.

    This is confounded by trace selection and is reported alongside, not in
    place of, the interventional estimate.
    """
    by_problem: dict = {}
    for pid, den, ans in traces:
        by_problem.setdefault(pid, []).append((den, ans))

    contrasts = []
    for pid, rows in by_problem.items():
        values = {d for d, _ in rows}
        for v in values:
            match = [a for d, a in rows if _eq(d, v)]
            other = [a for d, a in rows if not _eq(d, v)]
            if not match or not other:
                continue
            p_in = np.mean([_eq(a, v) for a in match])
            p_out = np.mean([_eq(a, v) for a in other])
            contrasts.append(p_in - p_out)
    return float(np.mean(contrasts)) if contrasts else float("nan")


# ---------------------------------------------------------------------------
# Partial identification and audit estimation (Propositions 2-4)
# ---------------------------------------------------------------------------

def identified_set(sigma: float, delta_bar: float = 1.0) -> tuple:
    """Sharp identified set for the follow rate given an observed sensitivity
    and a bound delta <= delta_bar (Proposition 2).

    Returns (lo, hi) = (sigma - min(sigma, delta_bar), sigma).
    With delta_bar = 1 this is [0, sigma]: sensitivity alone says nothing
    beyond the upper bound of Corollary 1.
    """
    m = min(sigma, delta_bar)
    return (sigma - m, sigma)


def comparison_identified_set(sigma1: float, sigma2: float,
                              delta_bar: float = 1.0) -> tuple:
    """Sharp identified set for phi_1 - phi_2 given both sensitivities and a
    common derailment bound (Corollary 2). Assumes sigma1 > sigma2."""
    m1, m2 = min(sigma1, delta_bar), min(sigma2, delta_bar)
    s = sigma1 - sigma2
    return (s - m1, s + m2)


def comparison_determined(sigma1: float, sigma2: float,
                          delta_bar: float = 1.0) -> bool:
    """True iff the sensitivity ordering determines the propagation ordering
    (Corollary 2): sigma1 - sigma2 > min(sigma1, delta_bar)."""
    if sigma1 <= sigma2:
        raise ValueError("Corollary 2 is stated for sigma1 > sigma2")
    return (sigma1 - sigma2) > min(sigma1, delta_bar)


def audit_estimate(sigma: float, audit_items: "Sequence[Item]") -> dict:
    """Audit-sample estimator of the follow rate (Proposition 3).

    sigma is taken from the full corpus, where it costs nothing beyond the
    corruption run. The evaluator is applied only to audit_items, which supply
    the derailment rate. Returns both estimators and their asymptotic
    variances so the efficiency comparison is explicit.
    """
    n = len(audit_items)
    if n == 0:
        return dict(n=0)
    labels = [classify(it) for it in audit_items]
    phi_dir = sum(l == FOLLOW for l in labels) / n
    delta_hat = sum(l == DERAIL for l in labels) / n
    phi_aud = sigma - delta_hat
    return dict(
        n=n,
        sigma=sigma,
        delta_hat=delta_hat,
        phi_direct=phi_dir,
        phi_audit=phi_aud,
        var_direct=phi_dir * (1 - phi_dir) / n,
        var_audit=delta_hat * (1 - delta_hat) / n,
        audit_preferred=delta_hat < phi_aud and sigma < 1.0,
    )


def variance_gap(phi: float, delta: float) -> tuple:
    """Proposition 3 identity. Returns (direct - audit, closed form), which
    must agree: phi(1-phi) - delta(1-delta) == (phi-delta)(1-sigma)."""
    sigma = phi + delta
    lhs = phi * (1 - phi) - delta * (1 - delta)
    rhs = (phi - delta) * (1 - sigma)
    return (lhs, rhs)


# ---------------------------------------------------------------------------
# Uncertainty: clustered bootstrap and permutation test
# ---------------------------------------------------------------------------

def cluster_bootstrap_ci(items: Sequence[Item],
                         statistic: Callable[[Sequence[Item]], float],
                         B: int = 10000,
                         alpha: float = 0.05,
                         seed: int = BASE_SEED) -> tuple:
    """Percentile bootstrap CI, resampling whole source problems.

    Items derived from the same source problem are correlated. Resampling
    items independently would understate the variance; resampling clusters
    does not.
    """
    rng = np.random.default_rng(seed)
    groups: dict = {}
    for it in items:
        groups.setdefault(it.problem_id, []).append(it)
    keys = list(groups.keys())
    if not keys:
        return (float("nan"), float("nan"), float("nan"))

    point = statistic(items)
    draws = np.empty(B)
    for b in range(B):
        picked = rng.integers(0, len(keys), size=len(keys))
        sample = [it for i in picked for it in groups[keys[i]]]
        draws[b] = statistic(sample)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return (point, float("nan"), float("nan"))
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(point), float(lo), float(hi))


def paired_permutation_p(diffs: np.ndarray,
                         clusters: Sequence | None = None,
                         B: int = 10000,
                         seed: int = BASE_SEED) -> float:
    """Two-sided sign-flip permutation test on paired differences.

    Under the null of no effect the differences are symmetric about zero, so
    flipping signs is a valid randomisation. Signs are flipped at the cluster
    level when clusters are supplied.
    """
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())

    if clusters is None:
        signs_shape = (B, diffs.size)
        flips = rng.choice([-1.0, 1.0], size=signs_shape)
        null = np.abs((flips * diffs).mean(axis=1))
    else:
        uniq = {c: i for i, c in enumerate(dict.fromkeys(clusters))}
        idx = np.array([uniq[c] for c in clusters])
        flips = rng.choice([-1.0, 1.0], size=(B, len(uniq)))
        null = np.abs((flips[:, idx] * diffs).mean(axis=1))

    return float((1.0 + np.sum(null >= obs)) / (B + 1.0))


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(name: str, items: Sequence[Item], B: int = 2000, seed: int = BASE_SEED) -> dict:
    r = outcome_rates(items)
    out = dict(model=name, **r)
    out["identity_residual"] = identity_residual(items)
    for key in ("phi", "delta", "sigma"):
        pt, lo, hi = cluster_bootstrap_ci(
            items, lambda s, k=key: outcome_rates(s)[k], B=B, seed=seed)
        out[f"{key}_ci"] = (lo, hi)
    if any(it.ans_placebo is not None for it in items):
        pt, lo, hi = cluster_bootstrap_ci(items, dpe, B=B, seed=seed)
        out["dpe"], out["dpe_ci"] = pt, (lo, hi)
    return out


def _eq(a, b) -> bool:
    """Answer-space equality. Exact for ints, strings and tuples; tolerant for
    floats. Replace this for domains with a richer value space."""
    if a is None or b is None:
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
        except (TypeError, ValueError):
            return False
    return a == b


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _synth(rule, n_problems=200, items_per_problem=3, seed=BASE_SEED,
           placebo_rate=0.0, copy_frac=0.0, incompetent_frac=0.0):
    """Build synthetic items from a planted answering rule.

    rule(rng, den_clean, den_treat) -> answer
    Values are distinct integers so that follow / restore / derail are
    unambiguous. A within-problem offset makes items correlated, which is what
    the clustered bootstrap is for.
    """
    rng = np.random.default_rng(seed)
    items = []
    for p in range(n_problems):
        for k in range(items_per_problem):
            dc = 1000 * p + 10 * k + 1
            dt = dc + 1
            ans = rule(rng, dc, dt)
            plc = (dc if rng.random() > placebo_rate else dt) if placebo_rate is not None else None
            items.append(Item(problem_id=f"p{p}", den_clean=dc, den_treat=dt,
                              ans_clean=dc, ans_treat=ans, ans_placebo=plc,
                              copyable=(rng.random() < copy_frac),
                              competent=(rng.random() >= incompetent_frac)))
    return items


def _check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    return bool(cond)


def smoke_test() -> int:
    print("denote_metrics smoke test (synthetic data, planted ground truth)")
    print("=" * 68)
    ok = True

    # T1 -- Proposition 1 holds exactly on gated samples, for arbitrary rules.
    print("\nT1  identity sigma == phi + delta on gated samples")
    worst = 0.0
    for s in range(20):
        rng0 = np.random.default_rng(s)
        p_f, p_r = rng0.random(), rng0.random()
        tot = p_f + p_r + rng0.random()
        pf, pr = p_f / tot, p_r / tot

        def rule(rng, dc, dt, pf=pf, pr=pr):
            u = rng.random()
            return dt if u < pf else (dc if u < pf + pr else dt + 7)

        gated = apply_gate(_synth(rule, seed=s))
        worst = max(worst, identity_residual(gated))
    ok &= _check("residual is zero up to floating point across 20 random rules",
                 worst < 1e-12, f"max residual = {worst:.3e} (exact in rational arithmetic; "
                                f"the counts are integers divided by n)")

    # T2 -- Corollary 1: sensitivity ranking can invert follow-rate ranking.
    print("\nT2  Corollary 1 realised: equal-or-higher sigma, lower phi")
    def make_rule(sigma0, phi0):
        def rule(rng, dc, dt):
            u = rng.random()
            if u < phi0:
                return dt                    # follow
            if u < sigma0:
                return dt + 7                # derail (changed, but not to v_e)
            return dc                        # restore
        return rule

    A = apply_gate(_synth(make_rule(0.80, 0.10), seed=1, n_problems=400))
    Bm = apply_gate(_synth(make_rule(0.50, 0.40), seed=2, n_problems=400))
    ra, rb = outcome_rates(A), outcome_rates(Bm)
    print(f"      model A: sigma={ra['sigma']:.3f}  phi={ra['phi']:.3f}  delta={ra['delta']:.3f}")
    print(f"      model B: sigma={rb['sigma']:.3f}  phi={rb['phi']:.3f}  delta={rb['delta']:.3f}")
    ok &= _check("A recovers planted (sigma=0.80, phi=0.10)",
                 abs(ra["sigma"] - 0.80) < 0.03 and abs(ra["phi"] - 0.10) < 0.03)
    ok &= _check("B recovers planted (sigma=0.50, phi=0.40)",
                 abs(rb["sigma"] - 0.50) < 0.03 and abs(rb["phi"] - 0.40) < 0.03)
    ok &= _check("ranking inverts: sigma(A)>sigma(B) but phi(A)<phi(B)",
                 ra["sigma"] > rb["sigma"] and ra["phi"] < rb["phi"])

    # T3 -- copy predicate.
    print("\nT3  copy-explicability predicate (C3)")
    ok &= _check("target present in literals is flagged",
                 is_copy_explicable(24, [3, 7, 24, 5]) is True)
    ok &= _check("target absent from literals is not flagged",
                 is_copy_explicable(31, [3, 7, 24, 5]) is False)
    ok &= _check("gate drops copyable items",
                 len(apply_gate([Item("p", 1, 2, 1, 2, None, copyable=True)])) == 0)

    # T4 -- DPE recovers a planted effect.
    print("\nT4  DPE recovers a planted contrast")
    true_follow, true_leak = 0.60, 0.15
    rng = np.random.default_rng(7)
    items = []
    for p in range(500):
        dc, dt = 10 * p + 1, 10 * p + 2
        at = dt if rng.random() < true_follow else dc
        ap = dt if rng.random() < true_leak else dc
        items.append(Item(f"p{p}", dc, dt, dc, at, ap))
    est, lo, hi = cluster_bootstrap_ci(items, dpe, B=2000, seed=BASE_SEED)
    target = true_follow - true_leak
    print(f"      planted DPE = {target:.3f}   estimate = {est:.3f}   95% CI = [{lo:.3f}, {hi:.3f}]")
    ok &= _check("CI covers the planted DPE", lo <= target <= hi)

    # T5 -- clustered bootstrap is wider than the naive one under correlation.
    print("\nT5  clustering widens the interval when items are correlated")
    rng = np.random.default_rng(11)
    corr = []
    for p in range(150):
        shared = rng.random() < 0.5           # whole problem follows or not
        for k in range(6):
            dc, dt = 100 * p + k, 100 * p + k + 50
            corr.append(Item(f"p{p}", dc, dt, dc, dt if shared else dc, None))
    stat = lambda s: outcome_rates(s)["phi"]
    _, clo, chi = cluster_bootstrap_ci(corr, stat, B=2000, seed=BASE_SEED)
    flat = [Item(f"i{j}", it.den_clean, it.den_treat, it.ans_clean, it.ans_treat, None)
            for j, it in enumerate(corr)]
    _, nlo, nhi = cluster_bootstrap_ci(flat, stat, B=2000, seed=BASE_SEED)
    print(f"      clustered width = {chi - clo:.4f}   naive width = {nhi - nlo:.4f}")
    ok &= _check("clustered interval is wider", (chi - clo) > (nhi - nlo))

    # T6 -- permutation test calibration under the null.
    print("\nT6  permutation test is calibrated under the null")
    rng = np.random.default_rng(3)
    rejects = 0
    trials = 200
    for t in range(trials):
        d = rng.choice([-1.0, 0.0, 1.0], size=120)   # symmetric about zero
        if paired_permutation_p(d, B=400, seed=t) < 0.05:
            rejects += 1
    rate = rejects / trials
    print(f"      empirical type-I rate at alpha=0.05: {rate:.3f} over {trials} trials")
    ok &= _check("rate is within Monte Carlo tolerance of 0.05", rate < 0.12)

    # T7 -- Holm correction.
    print("\nT7  Holm step-down correction")
    adj = holm([0.01, 0.04, 0.03])
    print(f"      raw [0.01, 0.04, 0.03] -> adjusted {np.round(adj, 4).tolist()}")
    ok &= _check("adjusted values dominate raw and stay monotone in rank",
                 np.all(adj >= np.array([0.01, 0.04, 0.03])) and adj[1] >= adj[2])

    # T8 -- on-policy estimator recovers a planted association.
    print("\nT8  on-policy estimator on planted traces")
    rng = np.random.default_rng(5)
    traces = []
    for p in range(300):
        v0, v1 = 10 * p, 10 * p + 1
        for _ in range(8):
            den = v0 if rng.random() < 0.5 else v1
            ans = den if rng.random() < 0.7 else (v1 if den == v0 else v0)
            traces.append((f"p{p}", den, ans))
    obs = observational_dpe(traces)
    print(f"      DPE_obs = {obs:.3f}   (planted alignment 0.70, so expect ~0.40)")
    ok &= _check("estimator is positive and near the planted contrast",
                 0.25 < obs < 0.55)

    # T9 -- Proposition 3 variance identity, checked symbolically then by simulation.
    print("\nT9  audit-estimator variance identity and efficiency")
    worst = 0.0
    rng = np.random.default_rng(21)
    for _ in range(500):
        phi = rng.random(); delta = rng.random() * (1 - phi)
        lhs, rhs = variance_gap(phi, delta)
        worst = max(worst, abs(lhs - rhs))
    ok &= _check("phi(1-phi)-delta(1-delta) == (phi-delta)(1-sigma)",
                 worst < 1e-12, f"max |lhs-rhs| = {worst:.3e}")

    true_phi, true_delta, n_aud = 0.45, 0.10, 400
    sigma_true = true_phi + true_delta
    rng = np.random.default_rng(22)
    d_est, a_est = [], []
    for _ in range(600):
        u = rng.random(n_aud)
        F = (u < true_phi)
        D = (u >= true_phi) & (u < sigma_true)
        d_est.append(F.mean())
        a_est.append(sigma_true - D.mean())
    vd, va = np.var(d_est), np.var(a_est)
    pred_d = true_phi * (1 - true_phi) / n_aud
    pred_a = true_delta * (1 - true_delta) / n_aud
    print(f"      direct: empirical var {vd:.2e} vs predicted {pred_d:.2e}")
    print(f"      audit : empirical var {va:.2e} vs predicted {pred_a:.2e}")
    ok &= _check("both empirical variances match the predicted Bernoulli forms",
                 abs(vd - pred_d) < 0.35 * pred_d and abs(va - pred_a) < 0.35 * pred_a)
    ok &= _check("audit estimator is more efficient when delta < phi", va < vd)

    # T10 -- Proposition 2: identified-set endpoints are attained.
    print("\nT10 sharp identified set endpoints are attainable")
    sigma0, dbar = 0.6, 0.25
    lo, hi = identified_set(sigma0, dbar)
    print(f"      sigma={sigma0}, delta_bar={dbar} -> identified set [{lo:.2f}, {hi:.2f}]")
    built = []
    for target_delta in (0.0, min(sigma0, dbar)):
        rng = np.random.default_rng(31)
        items = []
        for p in range(4000):
            dc, dt = 10 * p + 1, 10 * p + 2
            u = rng.random()
            a = dt if u < sigma0 - target_delta else (dt + 7 if u < sigma0 else dc)
            items.append(Item(f"p{p}", dc, dt, dc, a))
        built.append(outcome_rates(items)["phi"])
    print(f"      realised follow rates: {built[1]:.3f} (lower end), {built[0]:.3f} (upper end)")
    ok &= _check("both endpoints realised within sampling error",
                 abs(built[1] - lo) < 0.02 and abs(built[0] - hi) < 0.02)
    ok &= _check("with no bound on derailment no comparison is determined",
                 comparison_determined(0.8, 0.5, delta_bar=1.0) is False)
    ok &= _check("with a tight bound the same comparison is determined",
                 comparison_determined(0.8, 0.5, delta_bar=0.05) is True)

    print("\n" + "=" * 68)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    print("Note: these checks validate the estimators against planted values.")
    print("They are not evidence for any empirical claim in the paper.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(smoke_test())
