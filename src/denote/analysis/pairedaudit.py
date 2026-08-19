#!/usr/bin/env python3
"""denote_pairedaudit.py -- the paired discordance-bound audit of Supplement
A.5, Corollary 1 (cor:paired), recomputed on the released raw records.

Cited by name in denote_tdsc.tex ("Derailment does not pool across systems")
but never shipped with this bundle -- this is that script, written fresh
against the current (post-2026-08-14 B1 fix) raw records. Standard library
only, no imports from the rest of the bundle, same style as
denote_tdsc_verify.py: this is meant to be an independent second opinion,
not a second call to the same code.

For every pair of headline models (n >= 10, six models, 15 pairs):
  1. Join their gated records by record_id (same problem_id:seed:edit-step,
     i.e. the same source item and edit depth) -- "the items each pair
     shares" in the manuscript's own words.
  2. Discordance rate Pr[D1 != D2] on that shared population, exact
     one-sided Clopper-Pearson upper bound dbar.
  3. Independence-excess check: predicted rate delta1+delta2-2*delta1*delta2
     (deltas measured ON THE SHARED ITEMS) minus the observed rate, with a
     problem_id-clustered bootstrap (B=10,000) significance flag.
  4. Certification counts: g > m1 (unpaired, per-system bound), g > dbar
     (paired bound alone), g > min(m1, dbar) (the combined rule the paper
     recommends -- compute both, take the smaller).

MODES
  (no flags)   Print the full pairwise table and summary counts.
"""

import json
import math
import os
import random

RAW_DIR_DEFAULT = os.path.join("results", "raw")

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

# Headline panel and its per-system audit bound m1, exactly as printed in
# Table I (design-effect-corrected exact Clopper-Pearson construction, the
# primary specification since the 2026-08-14 promotion). Kept in sync with
# denote_tdsc_verify.py's MANUSCRIPT_TABLE1.
HEADLINE = {
    "Llama-70B": 0.190,
    "Llama-8B": 0.145,
    "Qwen-72B": 0.084,
    "Qwen-7B": 0.086,
    "Claude-5": 0.003,
    "GPT-5.6": 0.000,
}

TOL = 1e-9


def close(a, b, tol=TOL):
    return abs(a - b) <= tol


def gated_labels(records):
    """Apply C1-C2 gating and return {record_id: (derailed, sigma_label)}.

    derailed is 1 iff the record's treat answer is neither den_treat
    (follow) nor den_clean (restore) -- the manuscript's derailment
    condition, matching denote_tdsc_verify.py's gate_and_score. A null
    treat answer (no-response) is scored as derailment, the V2 convention.
    """
    out = {}
    for r in records:
        if r.get("status") != "built":
            continue
        ans = r.get("answers") or {}
        den_clean = r.get("den_clean")
        if ans.get("control") is None or den_clean is None:
            continue
        if not close(ans["control"], den_clean):
            continue
        at = ans.get("treat")
        if at is None:
            out[r["record_id"]] = (1, r.get("problem_id"))
            continue
        if close(at, r["den_treat"]):
            out[r["record_id"]] = (0, r.get("problem_id"))
        elif close(at, r["den_clean"]):
            out[r["record_id"]] = (0, r.get("problem_id"))
        else:
            out[r["record_id"]] = (1, r.get("problem_id"))
    return out


def clopper_pearson_upper(c, n, alpha=0.05):
    """Exact one-sided upper confidence limit for a binomial rate.

    Copied from denote_tdsc_verify.py rather than imported, per that
    script's own stated convention of not importing across the family.
    """
    if n <= 0:
        return 1.0
    if c >= n:
        return 1.0
    if c == 0:
        return 1.0 - alpha ** (1.0 / n)

    def tail(p):
        total = 0.0
        for k in range(c + 1):
            total += math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        return total

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if tail(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def load_labels(raw_dir):
    out = {}
    for fn in sorted(os.listdir(raw_dir)):
        if not (fn.startswith("full_") and fn.endswith(".json")):
            continue
        if "_grammar" in fn:
            continue
        with open(os.path.join(raw_dir, fn), encoding="utf-8") as fh:
            blob = json.load(fh)
        name = SHORT.get(blob["model"], blob["model"])
        if name not in HEADLINE:
            continue
        out[name] = gated_labels(blob["records"])
    return out


def pair_stats(labels_a, labels_b, rng):
    shared = sorted(set(labels_a) & set(labels_b))
    n = len(shared)
    if n == 0:
        return None

    d1 = [labels_a[rid][0] for rid in shared]
    d2 = [labels_b[rid][0] for rid in shared]
    probs = [labels_a[rid][1] for rid in shared]
    disc = [int(a != b) for a, b in zip(d1, d2)]

    n_disc = sum(disc)
    dbar = clopper_pearson_upper(n_disc, n)

    delta1 = sum(d1) / n
    delta2 = sum(d2) / n
    predicted = delta1 + delta2 - 2 * delta1 * delta2
    observed = n_disc / n
    excess = predicted - observed

    # problem_id-clustered bootstrap, B=10,000
    by_problem = {}
    for rid, a, b, p in zip(shared, d1, d2, probs):
        by_problem.setdefault(p, []).append((a, b))
    problem_ids = list(by_problem)
    B = 10_000
    excesses = []
    for _ in range(B):
        sample_a, sample_b = [], []
        for _ in problem_ids:
            p = rng.choice(problem_ids)
            for a, b in by_problem[p]:
                sample_a.append(a)
                sample_b.append(b)
        m = len(sample_a)
        if m == 0:
            continue
        sd1 = sum(sample_a) / m
        sd2 = sum(sample_b) / m
        spred = sd1 + sd2 - 2 * sd1 * sd2
        sobs = sum(int(x != y) for x, y in zip(sample_a, sample_b)) / m
        excesses.append(spred - sobs)
    excesses.sort()
    lo_ci = excesses[int(0.025 * len(excesses))]
    hi_ci = excesses[int(0.975 * len(excesses))]
    significant = not (lo_ci <= 0.0 <= hi_ci)

    return dict(n=n, n_disc=n_disc, dbar=dbar, delta1=delta1, delta2=delta2,
                predicted=predicted, observed=observed, excess=excess,
                lo_ci=lo_ci, hi_ci=hi_ci, significant=significant)


def main():
    raw_dir = RAW_DIR_DEFAULT
    labels = load_labels(raw_dir)
    names = sorted(HEADLINE)
    missing = [n for n in names if n not in labels]
    if missing:
        raise SystemExit(f"missing raw records for: {missing}")

    # sigma per model, over each model's OWN full gated population, to get g
    sigma = {n: sum(v[0] for v in labels[n].values()) / len(labels[n])
              for n in names}
    order = sorted(names, key=lambda n: -sigma[n])

    rng = random.Random(0)
    rows = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = order[i], order[j]
            g = abs(sigma[a] - sigma[b])
            st = pair_stats(labels[a], labels[b], rng)
            if st is None:
                continue
            m1 = min(HEADLINE[a], HEADLINE[b])
            rows.append((a, b, g, m1, st))

    print("Paired discordance-bound audit (Supplement A.5, Corollary 1)\n")
    print(f"  {'pair':<26}{'n':>5}{'g':>8}{'dbar':>8}{'m1':>8}"
          f"{'excess':>9}{'sig':>6}{'unpaired':>10}{'paired':>8}{'min':>6}")
    n_unpaired = n_paired = n_min = 0
    excesses = []
    for a, b, g, m1, st in rows:
        u_ok = g > m1
        p_ok = g > st["dbar"]
        m_ok = g > min(m1, st["dbar"])
        n_unpaired += u_ok
        n_paired += p_ok
        n_min += m_ok
        excesses.append(st["excess"])
        print(f"  {a:<12}-> {b:<11}{st['n']:>5}{g:>8.3f}{st['dbar']:>8.3f}"
              f"{m1:>8.3f}{st['excess']:>9.3f}"
              f"{'YES' if st['significant'] else 'no':>6}"
              f"{'clears' if u_ok else 'fails':>10}"
              f"{'clears' if p_ok else 'fails':>8}"
              f"{'clears' if m_ok else 'fails':>6}")

    print(f"\n  {len(rows)} pairs total")
    print(f"  unpaired bound (g > m1) certifies      {n_unpaired} of {len(rows)}")
    print(f"  paired bound   (g > dbar) certifies     {n_paired} of {len(rows)}")
    print(f"  combined min   (g > min(m1,dbar))       {n_min} of {len(rows)}")
    print(f"\n  independence-excess: mean {sum(excesses)/len(excesses):+.3f}, "
          f"range {min(excesses):+.3f} to {max(excesses):+.3f}")
    n_sig = sum(1 for _, _, _, _, st in rows if st["significant"])
    print(f"  significantly nonzero (clustered bootstrap, B=10,000): "
          f"{n_sig} of {len(rows)}")


if __name__ == "__main__":
    main()
