#!/usr/bin/env python3
"""denote_tdsc_verify.py -- independent verification of every quantity the
DENOTE manuscript reports, recomputed from the raw records rather than from any
summary file.

Written for the IEEE TDSC special-issue submission. Deliberately imports
nothing from the rest of the bundle: the point is to be a second opinion, not a
second call to the same code. Standard library only (math, json, itertools);
numpy is used only if present and is never required.

MODES
  --provenance   Reproduce the principal blocking finding: the manuscript's
                 headline table does not follow from the released records.
  --legacy       Check whether the fifth (legacy) arm was ever scored.
  --rank         Recompute determined pairs, adjacent transfers and admissible
                 rankings, under the sharp interval-order rule of the paper.
  --audit        Exact Clopper-Pearson audit sizes for a list of margins.
  --selftest     Run the synthetic smoke test. Uses NO data and asserts no
                 empirical outcome; it checks that the estimators recover
                 quantities that were planted in synthetic records.
  --preflight    Scan the .tex sources for scaffolding that must not ship.
  --all          Everything above.

EXIT CODE
  0 if every check that was run passed, 1 otherwise. A non-zero exit from
  --provenance or --legacy is the EXPECTED result on today's bundle; it means
  the blocking findings are still open, not that the script is broken.
"""

import argparse
import json
import math
import os
import random
import re
import sys
from itertools import permutations

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

# What the manuscript prints in Table I. Kept here so a mismatch is loud.
# m is the design-effect-corrected exact Clopper-Pearson construction, the
# primary specification since the 2026-08-14 promotion (denote_tdsc.tex,
# Table I footnote and the "ranking is never unique" paragraph).
MANUSCRIPT_TABLE1 = {
    #            N     sigma   phi     delta   m
    "Llama-70B": (134, 0.388, 0.276, 0.112, 0.190),
    "Llama-8B": (109, 0.202, 0.138, 0.064, 0.145),
    "Qwen-72B": (165, 0.188, 0.164, 0.024, 0.084),
    "Qwen-7B": (105, 0.086, 0.057, 0.029, 0.086),
    "Claude-5": (293, 0.003, 0.003, 0.000, 0.003),
    "GPT-5.6": (341, 0.000, 0.000, 0.000, 0.000),
}
MANUSCRIPT_DOWNSTREAM = {"determined": 9, "adjacent": 1, "rankings": 20}

TOL = 1e-3


# --------------------------------------------------------------------------
# core estimators
# --------------------------------------------------------------------------

def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def gate_and_score(records, policy="worst"):
    """Apply C1-C2 to built records and score the treat arm.

    C1 is implied by status == 'built'. C2 is the clean-agreement condition:
    the control answer must equal the trace's own denotation. Extraction
    failures (a null treat answer) are scored as derailment, which is the
    manuscript's stated V2 convention.

    Returns a dict with n, sigma, phi, delta, psi and the raw counts.
    """
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
        at = (r.get("answers") or {}).get("treat")
        if at is None:
            no_response += 1
            if policy == "complete":       # drop: an empty response is missing
                continue                    # data, not evidence of derailment
            derail += 1                     # "worst": charge it to derailment
            problems.add(r.get("problem_id"))
            continue
        problems.add(r.get("problem_id"))
        if close(at, r["den_treat"]):
            follow += 1
        elif close(at, r["den_clean"]):
            restore += 1
        else:
            derail += 1

    n = follow + restore + derail
    if n == 0:
        return dict(n=0, sigma=float("nan"), phi=float("nan"),
                    delta=float("nan"), psi=float("nan"), follow=0, restore=0,
                    derail=0, no_response=no_response, nprob=0)
    return dict(n=n, sigma=(follow + derail) / n, phi=follow / n,
                delta=derail / n, psi=restore / n, follow=follow,
                restore=restore, derail=derail, no_response=no_response,
                nprob=len(problems))


def load_panel(raw_dir):
    """Read every full_*.json and return {short_name: scored dict}.

    Excludes *_grammar.json variants: they share the same `model` field as
    their canonical counterpart (e.g. deepseek/deepseek-r1), so without this
    filter the dict-by-model-name assignment below silently lets whichever
    file sorts last overwrite the other, conflating canonical and
    grammar-tagged populations under one label. Found during the 2026-08-11
    revision pass: this made --provenance report a DeepSeek-R1 population of
    182 (canonical-only C1-C2 gate is 16; 182 came from the grammar file
    clobbering it).
    """
    out = {}
    if not os.path.isdir(raw_dir):
        raise SystemExit(f"raw directory not found: {raw_dir}")
    for fn in sorted(os.listdir(raw_dir)):
        if not (fn.startswith("full_") and fn.endswith(".json")):
            continue
        if "_grammar" in fn:
            continue
        with open(os.path.join(raw_dir, fn), encoding="utf-8") as fh:
            blob = json.load(fh)
        name = SHORT.get(blob["model"], blob["model"])
        out[name] = gate_and_score(blob["records"])
        out[name]["file"] = fn
    return out


def clopper_pearson_upper(c, n, alpha=0.05):
    """Exact one-sided upper confidence limit for a binomial rate.

    Solved by bisection on the tail probability rather than via scipy, so the
    script has no third-party dependency. At c = 0 this returns 1 - alpha**(1/n)
    in closed form, which is the case the paper's audit-size result uses.
    """
    if n <= 0:
        return 1.0
    if c >= n:
        return 1.0
    if c == 0:
        return 1.0 - alpha ** (1.0 / n)

    def tail(p):
        # P[X <= c] under Binomial(n, p); the limit solves this == alpha
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


def audit_size(g, alpha=0.05):
    """Smallest n with zero observed derailments that certifies margin g."""
    if g <= 0:
        return None
    n = 1
    while clopper_pearson_upper(0, n, alpha) >= g:
        n += 1
        if n > 10 ** 6:
            return None
    return n


# --------------------------------------------------------------------------
# ranking analysis (sharp interval-order rule, Proposition 4 of the paper)
# --------------------------------------------------------------------------

def rank_analysis(rows):
    """rows: list of (name, sigma, m). Returns determined pairs, adjacent
    transfers and the number of admissible rankings."""
    rows = sorted(rows, key=lambda r: -r[1])
    names = [r[0] for r in rows]
    sig = {r[0]: r[1] for r in rows}
    low = {r[0]: r[1] - r[2] for r in rows}          # t_i = sigma_i - m_i

    forced = set()
    for i in names:
        for j in names:
            if i != j and sig[j] < low[i]:
                forced.add((i, j))

    adjacent = [(names[k], names[k + 1]) for k in range(len(names) - 1)]
    adjacent_ok = [p for p in adjacent if p in forced]

    admissible = 0
    for perm in permutations(names):
        pos = {n: r for r, n in enumerate(perm)}
        if all(pos[i] < pos[j] for i, j in forced):
            admissible += 1

    return dict(order=names, forced=len(forced),
                pairs=len(names) * (len(names) - 1) // 2,
                adjacent_ok=len(adjacent_ok), adjacent=len(adjacent),
                admissible=admissible, total=math.factorial(len(names)),
                adjacent_detail=[(a, b, sig[a] - sig[b], sig[a] - low[a],
                                  (a, b) in forced) for a, b in adjacent])


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def mode_provenance(raw_dir):
    panel = load_panel(raw_dir)
    print("Recomputed from the released raw records "
          "(C1-C2 gating, extraction failures scored as derailment):\n")
    print(f"  {'model':<12}{'n':>5}{'sigma':>9}{'phi':>9}{'delta':>9}"
          f"{'no-resp':>9}")
    total_nr = 0
    for name in sorted(panel):
        s = panel[name]
        print(f"  {name:<12}{s['n']:>5}{s['sigma']:>9.3f}{s['phi']:>9.3f}"
              f"{s['delta']:>9.3f}{s['no_response']:>9}")
        total_nr += s["no_response"]

    print("\nAgainst the manuscript's Table I:\n")
    ok = True
    for name, (N, sigma, phi, delta, _m) in MANUSCRIPT_TABLE1.items():
        s = panel.get(name)
        if s is None:
            print(f"  {name:<12} NOT FOUND in raw records")
            ok = False
            continue
        bad = []
        if s["n"] != N:
            bad.append(f"n {s['n']} vs {N}")
        if abs(s["sigma"] - sigma) > TOL:
            bad.append(f"sigma {s['sigma']:.3f} vs {sigma:.3f}")
        if abs(s["phi"] - phi) > TOL:
            bad.append(f"phi {s['phi']:.3f} vs {phi:.3f}")
        if abs(s["delta"] - delta) > TOL:
            bad.append(f"delta {s['delta']:.3f} vs {delta:.3f}")
        if bad:
            ok = False
            print(f"  {name:<12} MISMATCH: " + "; ".join(bad))
        else:
            print(f"  {name:<12} matches")

    print(f"\n  treat-arm no-response calls in the gated population: {total_nr}")
    if not ok:
        print("\n  BLOCKING (item B1). The manuscript's headline table does not\n"
              "  follow from the released records. Its numbers are computed\n"
              "  after re-running the failed calls, and those post-re-run\n"
              "  records are not in this bundle. Ship them, or revert the\n"
              "  paper's numbers to what is reproducible here.")
    return ok


def mode_legacy(raw_dir):
    print("Legacy arm: prompts constructed vs outputs actually scored\n")
    ok = True
    for fn in sorted(os.listdir(raw_dir)):
        if not (fn.startswith("full_") and fn.endswith(".json")):
            continue
        with open(os.path.join(raw_dir, fn), encoding="utf-8") as fh:
            blob = json.load(fh)
        built = [r for r in blob["records"] if r.get("status") == "built"]
        prompts = sum(1 for r in built if "legacy" in (r.get("conditions") or {}))
        outputs = sum(1 for r in built if "legacy" in (r.get("outputs") or {}))
        answers = sum(1 for r in built if "legacy" in (r.get("answers") or {}))
        flag = "" if outputs else "   <-- never scored"
        print(f"  {SHORT.get(blob['model'], blob['model']):<14}"
              f"built={len(built):>5}  prompts={prompts:>5}  "
              f"outputs={outputs:>5}  answers={answers:>5}{flag}")
        if outputs == 0 and prompts > 0:
            ok = False
    if not ok:
        print("\n  BLOCKING (item B3). Legacy prompts exist and legacy outputs do\n"
              "  not, on every model. Any legacy head-to-head number has no\n"
              "  artifact behind it. Score the arm or delete the claim.")
    return ok


def mode_rank(raw_dir, summary_path):
    print("Ranking analysis under the sharp interval-order rule.\n")

    rows_paper = [(k, v[1], v[4]) for k, v in MANUSCRIPT_TABLE1.items()]
    res = rank_analysis(rows_paper)
    print("  (a) from the manuscript's Table I as printed")
    print(f"      determined {res['forced']}/{res['pairs']}   "
          f"adjacent {res['adjacent_ok']}/{res['adjacent']}   "
          f"admissible {res['admissible']}/{res['total']}")
    internal_ok = (res["forced"] == MANUSCRIPT_DOWNSTREAM["determined"]
                   and res["adjacent_ok"] == MANUSCRIPT_DOWNSTREAM["adjacent"]
                   and res["admissible"] == MANUSCRIPT_DOWNSTREAM["rankings"])
    print("      internally consistent with the text: "
          f"{'YES' if internal_ok else 'NO'}")

    if not os.path.exists(summary_path):
        print(f"\n  (b) skipped: {summary_path} not found")
        return internal_ok

    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    panel = load_panel(raw_dir)
    rows_shipped, seen = [], set()
    for m in summary.get("models", []):
        name = SHORT.get(m["model"])
        if name is None or name in seen:
            continue
        if panel.get(name, {}).get("n", 0) < 10:
            continue
        seen.add(name)
        rows_shipped.append((name, m["sigma"], min(m["sigma"], m["delta_bar"])))

    res2 = rank_analysis(rows_shipped)
    print("\n  (b) from the SHIPPED analysis output (pre-re-run)")
    print(f"      determined {res2['forced']}/{res2['pairs']}   "
          f"adjacent {res2['adjacent_ok']}/{res2['adjacent']}   "
          f"admissible {res2['admissible']}/{res2['total']}")
    print(f"      sigma ordering: {' > '.join(res2['order'])}")
    for a, b, g, m1, okp in res2["adjacent_detail"]:
        print(f"        {a:<11}-> {b:<11} g={g:.3f}  m1={m1:.3f}  "
              f"{'clears' if okp else 'FAILS'}")
    if (res["admissible"], res["forced"]) != (res2["admissible"], res2["forced"]):
        print("\n  The two panels disagree. See item B1.")
    return internal_ok


def mode_grid(raw_dir):
    """Reproduce Table III: admissible rankings across the two analysis choices."""
    head = ["Llama-70B", "Llama-8B", "Qwen-72B", "Qwen-7B", "Claude-5", "GPT-5.6"]
    scored = {}
    for fn in sorted(os.listdir(raw_dir)):
        if not (fn.startswith("full_") and fn.endswith(".json")):
            continue
        if "_grammar" in fn:  # see load_panel() docstring: same-model clobber risk
            continue
        with open(os.path.join(raw_dir, fn), encoding="utf-8") as fh:
            blob = json.load(fh)
        name = SHORT.get(blob["model"], blob["model"])
        scored[name] = {pol: gate_and_score(blob["records"], pol)
                        for pol in ("worst", "complete")}

    print("Admissible rankings across analysis choices "
          "(reproducible rows only).\n")
    print(f"  {'empty responses':<16}{'delta-bar from':<28}"
          f"{'det.':>6}{'adj.':>6}{'rankings':>10}")
    counts = []
    for pol, plabel in (("complete", "complete case"), ("worst", "worst case")):
        for corr, clabel in ((False, "exact CP"),
                             (True, "exact CP + design effect")):
            rows = []
            for name in head:
                s_ = scored[name][pol]
                if corr:
                    deff = s_["n"] / s_["nprob"] if s_["nprob"] else 1.0
                    c, n = max(0, round(s_["derail"] / deff)), max(1, round(s_["n"] / deff))
                else:
                    c, n = s_["derail"], s_["n"]
                dbar = clopper_pearson_upper(c, n)
                rows.append((name, s_["sigma"], min(s_["sigma"], dbar)))
            res = rank_analysis(rows)
            counts.append(res["admissible"])
            print(f"  {plabel:<16}{clabel:<28}{res['forced']:>6}"
                  f"{res['adjacent_ok']:>6}{res['admissible']:>10}")
    print(f"\n  range {min(counts)} to {max(counts)} of 720; "
          f"unique under any row: {'YES' if 1 in counts else 'NO'}")
    print("  excluded fraction runs "
          f"{100 * (1 - max(counts) / 720):.1f}% to {100 * (1 - min(counts) / 720):.1f}%")
    return 1 not in counts


def mode_audit():
    print("Exact Clopper-Pearson audit sizes at alpha = 0.05, zero derailments "
          "observed.\n")
    print(f"  {'margin g':>10}{'exact n':>10}{'ceil(3/g)':>12}")
    for g in (0.186, 0.102, 0.082, 0.043, 0.014, 0.010, 0.002):
        n = audit_size(g)
        print(f"  {g:>10.3f}{n:>10}{math.ceil(3 / g):>12}")
    # The paper's claim: the 3/g rule is an upper envelope for alpha = 0.05.
    ok = all(audit_size(g) <= math.ceil(3 / g)
             for g in (0.186, 0.102, 0.082, 0.043, 0.014, 0.010, 0.002))
    print(f"\n  3/g dominates the exact size at every margin tested: "
          f"{'YES' if ok else 'NO'}")
    return ok


def mode_selftest():
    """Synthetic smoke test. Plants known rates in fabricated records, then
    checks the estimators recover them. It computes; it does not assert any
    empirical result about the real panel."""
    print("Synthetic smoke test (no data read, no empirical claim asserted).\n")
    rng = random.Random(20260809)
    ok = True

    # 1. estimator recovery
    n, want_phi, want_delta = 4000, 0.30, 0.12
    recs = []
    for _ in range(n):
        u = rng.random()
        if u < want_phi:
            treat = 2.0                      # follow: equals den_treat
        elif u < want_phi + want_delta:
            treat = 99.0                     # derail: neither denotation
        else:
            treat = 1.0                      # restore: equals den_clean
        recs.append(dict(status="built", den_clean=1.0, den_treat=2.0,
                         answers=dict(control=1.0, treat=treat)))
    got = gate_and_score(recs)
    for label, want, have in (("phi", want_phi, got["phi"]),
                              ("delta", want_delta, got["delta"])):
        good = abs(have - want) < 0.03
        ok &= good
        print(f"  recover {label:<6} planted {want:.3f} -> "
              f"estimated {have:.3f}   {'PASS' if good else 'FAIL'}")

    # 2. the identity the paper checks in every table row
    ident = abs(got["sigma"] - (got["phi"] + got["delta"])) < 1e-12
    ok &= ident
    print(f"  identity sigma = phi + delta holds exactly            "
          f"{'PASS' if ident else 'FAIL'}")

    # 3. C2 gating actually removes items
    recs2 = recs + [dict(status="built", den_clean=1.0, den_treat=2.0,
                         answers=dict(control=7.0, treat=2.0))] * 50
    gated_ok = gate_and_score(recs2)["n"] == n
    ok &= gated_ok
    print(f"  C2 drops the 50 planted clean-disagreement items        "
          f"{'PASS' if gated_ok else 'FAIL'}")

    # 4. extraction failures are scored as derailment, not dropped
    recs3 = [dict(status="built", den_clean=1.0, den_treat=2.0,
                  answers=dict(control=1.0, treat=None))] * 10
    s3 = gate_and_score(recs3)
    conv_ok = s3["n"] == 10 and abs(s3["delta"] - 1.0) < 1e-12
    ok &= conv_ok
    print(f"  null treat answers score as derailment (V2 convention)  "
          f"{'PASS' if conv_ok else 'FAIL'}")

    # 5. Clopper-Pearson closed form at c = 0
    cp_ok = abs(clopper_pearson_upper(0, 30) - (1 - 0.05 ** (1 / 30))) < 1e-12
    ok &= cp_ok
    print(f"  Clopper-Pearson c=0 matches 1-alpha^(1/n)               "
          f"{'PASS' if cp_ok else 'FAIL'}")

    # 6. interval order: a planted forced pair must be respected
    res = rank_analysis([("A", 0.50, 0.10), ("B", 0.20, 0.05), ("C", 0.05, 0.05)])
    order_ok = res["forced"] == 3 and res["admissible"] == 1
    ok &= order_ok
    print(f"  fully separated intervals give a unique ranking         "
          f"{'PASS' if order_ok else 'FAIL'}")

    # 7. and an unbounded case must forbid nothing
    res2 = rank_analysis([("A", 0.50, 0.50), ("B", 0.20, 0.20), ("C", 0.05, 0.05)])
    vac_ok = res2["forced"] == 0 and res2["admissible"] == 6
    ok &= vac_ok
    print(f"  unbounded derailment leaves every ordering admissible   "
          f"{'PASS' if vac_ok else 'FAIL'}")

    print(f"\n  smoke test: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return ok


def mode_certify():
    """Run the certification, family-wise and evasion experiments (E1-E3).
    Delegates to denote_tdsc_certify.py so there is exactly one implementation."""
    import subprocess
    r = subprocess.run([sys.executable, "denote_tdsc_certify.py"],
                       capture_output=True, text=True)
    print(r.stdout or r.stderr)
    return r.returncode == 0


SCAFFOLD_PATTERNS = [
    (r"\\rf\{", "red flag \\rf{...}"),
    (r"\\rx\{", "TO RUN marker \\rx{...}"),
    (r"\\rp\{", "ARTIFACT MISSING marker \\rp{...}"),
    (r"\\rc\b", "empty-cell marker \\rc"),
    (r"\[RESULT PLACEHOLDER", "result placeholder"),
    (r"\[CITATION NEEDED", "citation needed"),
    (r"\[UNVERIFIED\]", "unverified tag"),
    (r">>> TO RUN", "to-run block"),
    (r"MISSING AUTHORS", "bibliography with missing authors"),
    (r"\\drafttrue", "draft switch still true"),
]


def mode_preflight(tex_files):
    print("Preflight scan of the LaTeX sources.\n")
    clean = True
    for path in tex_files:
        if not os.path.exists(path):
            print(f"  {path}: NOT FOUND")
            clean = False
            continue
        text = open(path, encoding="utf-8").read()
        # ignore the macro definitions themselves and comment lines
        body = "\n".join(l for l in text.splitlines()
                         if not l.lstrip().startswith("%")
                         and "newcommand" not in l)
        hits = []
        for pat, label in SCAFFOLD_PATTERNS:
            k = len(re.findall(pat, body))
            if k:
                hits.append(f"{label} x{k}")
        if hits:
            clean = False
            print(f"  {path}: NOT READY -- " + "; ".join(hits))
        else:
            print(f"  {path}: clean")
    if not clean:
        print("\n  Resolve every item in REDFLAGS_TDSC.txt, set \\draftfalse in\n"
              "  both sources, and run this again before submitting.")
    return clean


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=RAW_DIR_DEFAULT)
    ap.add_argument("--summary", default=os.path.join("results", "summary.json"))
    for flag in ("provenance", "legacy", "rank", "grid", "audit", "certify",
                 "selftest", "preflight", "all"):
        ap.add_argument(f"--{flag}", action="store_true")
    args = ap.parse_args()

    if not any([args.provenance, args.legacy, args.rank, args.grid, args.audit,
                args.certify, args.selftest, args.preflight, args.all]):
        args.all = True

    results = {}
    if args.selftest or args.all:
        results["selftest"] = mode_selftest()
        print()
    if args.provenance or args.all:
        results["provenance"] = mode_provenance(args.raw)
        print()
    if args.legacy or args.all:
        results["legacy"] = mode_legacy(args.raw)
        print()
    if args.rank or args.all:
        results["rank"] = mode_rank(args.raw, args.summary)
        print()
    if args.grid or args.all:
        results["grid"] = mode_grid(args.raw)
        print()
    if args.audit or args.all:
        results["audit"] = mode_audit()
        print()
    if args.certify or args.all:
        results["certify"] = mode_certify()
        print()
    if args.preflight or args.all:
        results["preflight"] = mode_preflight(
            ["denote_tdsc.tex", "denote_tdsc_supp.tex"])
        print()

    print("=" * 64)
    for k, v in results.items():
        print(f"  {k:<12} {'PASS' if v else 'OPEN / FAIL'}")
    print("=" * 64)
    if not results.get("selftest", True):
        print("The smoke test failed, so no other result in this run is "
              "trustworthy. Fix the script first.")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
