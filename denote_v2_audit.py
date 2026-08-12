#!/usr/bin/env python3
"""denote_v2_audit.py -- the V2 extraction audit. THIS IS THE BLOCKING ITEM.

Every identification claim in the paper rests on delta. Right now an item is
scored as derailed whenever the extracted answer is neither [[T]] nor [[T^e]],
which silently counts EXTRACTION FAILURES as derailment. If a material share
of derailments are really the extractor missing an answer that is present in
the text, then delta is overstated, m is overstated, and every identified set
in Table 1 and Figure 1 is too wide.

The audit is two phases with a human in the middle.

  phase 1   python denote_v2_audit.py --sample --per-model 50
            Draws derailed items, at most one per source problem so the
            Clopper-Pearson bound of Proposition 6 applies, and writes
            results/v2_audit_sheet.csv with the raw continuation text.

  phase 2   a human fills the `verdict` column with one of
                extraction   the answer IS in the text, the extractor missed it
                derail       the model really landed somewhere else
                unclear      cannot tell
            then:
            python denote_v2_audit.py --analyze

Phase 2 prints the extraction share with an exact interval, the corrected
delta per model, and the recomputed identified sets, and writes
results/v2_audit_result.json.

No number in this script is invented. If the sheet is unfilled it says so and
exits.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path

try:
    from scipy.stats import beta as _beta
except ImportError:
    _beta = None

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results"
SHEET = OUT / "v2_audit_sheet.csv"
RESULT = OUT / "v2_audit_result.json"
PILOT = ("qwen/qwen-2.5-7b-instruct", "2026-07-26")
VERDICTS = {"extraction", "derail", "unclear"}


def _eq(a, b, tol=1e-6):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return str(a) == str(b)


def load_states():
    """Every full run, excluding the pilot (see denote_fix_bugs.py)."""
    states = []
    for p in sorted(RAW.glob("full_*.json")):
        s = json.loads(p.read_text())
        if (s.get("model"), s.get("access_date")) == PILOT:
            continue
        states.append((p, s))
    if not states:
        sys.exit(f"no runs found under {RAW}")
    return states


def derailed_records(state):
    """Treat-arm items whose answer is neither the clean nor the edited value."""
    out = []
    for r in state.get("records", []):
        if r.get("status") != "built":
            continue
        ans = (r.get("answers") or {}).get("treat")
        if ans is None:
            continue        # no-response: handled by --prescreen, not by hand
        if _eq(ans, r.get("den_treat")) or _eq(ans, r.get("den_clean")):
            continue
        out.append(r)
    return out


def cp_upper(k, n, alpha=0.05):
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    if _beta is None:
        return min(1.0, 3.0 / n) if k == 0 else float("nan")
    return float(_beta.ppf(1 - alpha, k + 1, n - k))


def cp_lower(k, n, alpha=0.05):
    if n == 0 or k == 0:
        return 0.0
    if _beta is None:
        return float("nan")
    return float(_beta.ppf(alpha, k, n - k + 1))


# ----------------------------------------------------------------- phase 1
def do_sample(per_model, seed):
    import random
    rng = random.Random(seed)
    rows = []
    for path, state in load_states():
        model = state["model"]
        der = derailed_records(state)
        # one per source problem keeps the audit a binomial sample
        by_problem = {}
        for r in der:
            by_problem.setdefault(r.get("problem_id"), []).append(r)
        pids = sorted(by_problem)
        rng.shuffle(pids)
        picked = [rng.choice(by_problem[pid]) for pid in pids[:per_model]]
        for r in picked:
            rows.append({
                "model": model,
                "record_id": r.get("record_id"),
                "problem_id": r.get("problem_id"),
                "den_clean": r.get("den_clean"),
                "den_treat": r.get("den_treat"),
                "extracted_answer": (r.get("answers") or {}).get("treat"),
                "continuation_text": (r.get("outputs") or {}).get("treat", ""),
                "verdict": "",
                "note": "",
            })
        print(f"  {model:38s} derailed={len(der):4d} "
              f"problems={len(pids):4d} sampled={min(per_model, len(pids)):3d}")
    if not rows:
        sys.exit("no derailed items found; nothing to audit")
    OUT.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {SHEET}  ({len(rows)} rows)")
    print("Fill the `verdict` column with: extraction | derail | unclear")
    print("An item is `extraction` if the correct value is visible in the")
    print("continuation text but the extractor did not pick it up.")
    print("Then run:  python denote_v2_audit.py --analyze")


# ----------------------------------------------------------------- phase 2
def do_analyze():
    if not SHEET.exists():
        sys.exit(f"{SHEET} not found; run --sample first")
    rows = list(csv.DictReader(SHEET.open(encoding="utf-8")))
    filled = [r for r in rows if (r.get("verdict") or "").strip() in VERDICTS]
    if not filled:
        sys.exit(f"{SHEET} has no verdicts filled in. Nothing to analyse.")
    if len(filled) < len(rows):
        print(f"WARNING: {len(rows) - len(filled)} of {len(rows)} rows unfilled; "
              f"analysing the {len(filled)} that are done.\n")

    states = {s["model"]: s for _, s in load_states()}
    summary_path = OUT / "summary.json"
    summ = json.loads(summary_path.read_text()) if summary_path.exists() else {"models": []}
    prior = {m["model"]: m for m in summ.get("models", [])
             if (m.get("model"), m.get("access_date")) != PILOT}

    out = {"per_model": [], "note": "corrected delta assumes audited items are "
                                    "representative of all derailed items"}
    print(f"{'model':38s} {'n_a':>4s} {'extr':>5s} {'share':>16s} "
          f"{'delta':>7s} {'delta_corrected':>17s}")
    for model in sorted({r["model"] for r in filled}):
        sub = [r for r in filled if r["model"] == model]
        n = len(sub)
        k = sum(1 for r in sub if r["verdict"].strip() == "extraction")
        unclear = sum(1 for r in sub if r["verdict"].strip() == "unclear")
        lo, hi = cp_lower(k, n), cp_upper(k, n)
        share = k / n if n else float("nan")
        d_old = prior.get(model, {}).get("delta")
        d_new = d_old * (1 - share) if d_old is not None else None
        d_new_lo = d_old * (1 - hi) if d_old is not None else None
        d_new_hi = d_old * (1 - lo) if d_old is not None else None
        print(f"{model:38s} {n:4d} {k:5d} "
              f"{share:6.3f} [{lo:.3f},{hi:.3f}] "
              f"{(d_old if d_old is not None else float('nan')):7.3f} "
              f"{(d_new if d_new is not None else float('nan')):7.3f} "
              f"[{(d_new_lo if d_new_lo is not None else float('nan')):.3f},"
              f"{(d_new_hi if d_new_hi is not None else float('nan')):.3f}]")
        out["per_model"].append({
            "model": model, "n_audited": n, "extraction_failures": k,
            "unclear": unclear, "extraction_share": share,
            "extraction_share_ci": [lo, hi],
            "delta_reported": d_old, "delta_corrected": d_new,
            "delta_corrected_ci": [d_new_lo, d_new_hi],
        })
    worst = max((m["extraction_share"] for m in out["per_model"]), default=0.0)
    out["max_extraction_share"] = worst
    RESULT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULT}")
    print()
    if worst >= 0.20:
        print("VERDICT: extraction share reaches "
              f"{worst:.1%} for at least one model.")
        print("  delta is materially overstated. Recompute every delta, every m,")
        print("  every identified set, Table 1, Table 2 and Figure 1 before")
        print("  making any claim in the paper. Re-run denote_reconcile.py.")
    elif worst >= 0.05:
        print(f"VERDICT: extraction share up to {worst:.1%}. Report it in the")
        print("  paper and state the direction of the bias; the identified sets")
        print("  are conservative by roughly that factor.")
    else:
        print(f"VERDICT: extraction share below 5% everywhere ({worst:.1%}).")
        print("  Report the audit and its interval. V2 is discharged.")


def do_prescreen():
    """Automated first cut: separate NO-RESPONSE calls from real derailment.

    A call that returned an empty or truncated body is missing data, not a
    derailment. Scoring it as derailment inflates delta. This needs no human.
    """
    print("model                              N  no-resp  sigma    phi  delta "
          "| sigma*   phi* delta*")
    print("-" * 100)
    out = []
    for _, s in load_states():
        recs = [r for r in s.get("records", []) if r.get("status") == "built"]
        n = len(recs)
        if not n:
            continue
        F = R = D = E = 0
        for r in recs:
            a = (r.get("answers") or {}).get("treat")
            if a is None:
                E += 1
            elif _eq(a, r.get("den_treat")):
                F += 1
            elif _eq(a, r.get("den_clean")):
                R += 1
            else:
                D += 1
        sig, phi, dl = (F + D + E) / n, F / n, (D + E) / n
        m = n - E
        sig2 = (F + D) / m if m else 0.0
        phi2 = F / m if m else 0.0
        dl2 = D / m if m else 0.0
        print(f"{s['model'][:33]:33s} {n:4d} {E:8d} {sig:6.3f} {phi:6.3f} {dl:6.3f} "
              f"| {sig2:6.3f} {phi2:6.3f} {dl2:6.3f}")
        out.append({"model": s["model"], "n_built": n, "no_response": E,
                    "as_scored": {"sigma": sig, "phi": phi, "delta": dl},
                    "dropping_no_response": {"sigma": sig2, "phi": phi2, "delta": dl2},
                    "delta_entirely_no_response": dl > 0 and dl2 == 0})
    tot = sum(o["no_response"] for o in out)
    tn = sum(o["n_built"] for o in out)
    print()
    print(f"no-response calls: {tot} of {tn} built items ({tot/tn:.1%})")
    bad = [o["model"] for o in out if o["delta_entirely_no_response"]]
    if bad:
        print()
        print("MODELS WHOSE ENTIRE DERAILMENT RATE IS NO-RESPONSE CALLS:")
        for b in bad:
            print(f"   {b}")
        print()
        print("For these, delta is not a property of the model. Their identified")
        print("sets collapse to points and any claim that their band 'reaches")
        print("zero' is an artifact of dropped calls. Either re-run the failed")
        print("calls, or treat them as missing data and report a response rate.")
    (OUT / "v2_prescreen.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'v2_prescreen.json'}")
    print("Then audit the REMAINING derailments for extraction error: --sample")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prescreen", action="store_true",
                    help="automated: separate no-response calls from derailment")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--per-model", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260803)
    a = ap.parse_args()
    if a.prescreen:
        do_prescreen()
    elif a.sample:
        do_sample(a.per_model, a.seed)
    elif a.analyze:
        do_analyze()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
