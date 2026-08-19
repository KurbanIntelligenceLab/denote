#!/usr/bin/env python3
"""
denote_grammar_ext.py

A text-normalization layer in front of the existing, unmodified
``ArithTrace`` grammar (denote_build.py), motivated by reading the raw
DeepSeek-R1 rejections in ``results/raw/full_deepseek_deepseek-r1.json``.

Why a separate module rather than editing denote_build._STEP
---------------------------------------------------------------
The panel's headline numbers (Table 1, seeds 0-2) were computed against the
exact current ``_STEP`` regex. Changing that regex in place would silently
change what "built" means for data already collected and analysed. This
module instead rewrites trace TEXT into the format ``_STEP`` already
accepts, then hands it to the untouched ``ArithTrace.parse``. Applying it is
therefore an explicit, auditable choice at the call site, not a retroactive
change to existing results.

What reading the real rejections found
----------------------------------------
Of 404 rejected DeepSeek-R1 records (seeds 0-2 combined) in the raw JSON:
  - 99  have an entirely empty trace. This is a *sampling*-stage failure
        (the ``sample_trace`` completion came back empty), not a grammar
        problem, and a text normalizer cannot fix it. It also cannot be
        re-collected in this environment (no OPENROUTER_API_KEY). It is
        reported separately below and is NOT claimed as fixed here.
  - 157 have non-empty text where zero lines match ``_STEP``. Reading a
        sample shows free-form prose that is truncated (by
        ``max_trace_tokens``) before reaching any structured line at all --
        a token-budget / prompt-adherence problem, not a format the
        normalizer below can repair (there is no structured content to
        rewrite).
  - 128 DO have at least one parseable line but fail self-consistency (C2).
        Reading these shows two recurring, non-DeepSeek-specific format
        gaps against the documented grammar:
          (a) TRACE_SYSTEM explicitly permits introducing a constant with
              ``name = number`` (no operator), but ``_STEP`` requires an
              operator unconditionally, so every such line was silently
              dropped from the parsed DAG.
          (b) A common "show the substitution" style, e.g.
              ``total = a + b = 3 + 4 = 7``, has three ``=`` signs; ``_STEP``
              expects exactly two, so the whole line is dropped.
        Both are addressed by ``normalize_trace_text`` below by rewriting
        the line into the two-``=`` shape ``_STEP`` already parses, keeping
        the same final value.

Run the self-test (offline; also re-gates the real DeepSeek raw JSON if
present, reporting the retroactive C1/C2 improvement -- not a re-measurement
of phi/delta/sigma, since no new model call is made):
    python3 denote_grammar_ext.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from denote_build import ArithTrace, _ANSWER, _STEP, build_items_all_positions

_BARE_CONST = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z_0-9 ]*?)\s*=\s*(?P<val>-?\d+(?:\.\d+)?)\s*$"
)
# Matches a trailing, fully-numeric "lhs op rhs = val" clause anywhere in the
# line, so a "name = SYMBOLIC = numeric op numeric = val" substitution-style
# line collapses to the final, already-numeric clause. Non-greedy prefix so
# the LAST such clause wins if there is more than one "=" in the line.
_MULTI_EQ_TAIL = re.compile(
    r"^\s*(?:(?P<name>[A-Za-z_][A-Za-z_0-9 ]*?)\s*=\s*)?"
    r".*?(?P<lhs>-?\d+(?:\.\d+)?)\s*(?P<op>[-+*/])\s*(?P<rhs>-?\d+(?:\.\d+)?)"
    r"\s*=\s*(?P<val>-?\d+(?:\.\d+)?)\s*$"
)


def normalize_line(raw: str) -> str:
    """Rewrite one line into the ``_STEP``-parseable shape if it matches one
    of the two documented-but-unparseable patterns above; otherwise return
    it unchanged (the normalizer never invents content it cannot justify
    from the line itself).

    When: normalize_trace_text(), once per line.
    """
    if _STEP.match(raw) or _ANSWER.search(raw):
        return raw
    m = _BARE_CONST.match(raw)
    if m:
        val = m.group("val")
        return f"{m.group('name')} = {val} + 0 = {val}"
    m = _MULTI_EQ_TAIL.match(raw)
    if m:
        prefix = f"{m.group('name')} = " if m.group("name") else ""
        return f"{prefix}{m.group('lhs')} {m.group('op')} {m.group('rhs')} = {m.group('val')}"
    return raw


def normalize_trace_text(text: str) -> str:
    """Apply normalize_line to every line of a trace.

    When: as a drop-in pre-processing step before ArithTrace.parse /
    build_items_all_positions, at the call site that opts into it.
    """
    return "\n".join(normalize_line(line) for line in text.splitlines())


# ---------------------------------------------------------------------------
# Self-test: synthetic cases, then a real retroactive re-gate of the actual
# DeepSeek-R1 raw JSON if it is present (offline; no model calls)
# ---------------------------------------------------------------------------

def _check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    return bool(cond)


def self_test() -> int:
    print("denote_grammar_ext self-test")
    print("=" * 66)
    ok = True

    print("\nG1  bare-constant lines (permitted by TRACE_SYSTEM, unparseable by _STEP)")
    ok &= _check("'file_size = 200' does not parse under the base grammar",
                ArithTrace.parse("file_size = 200\nThe final answer is 200") is None)
    fixed = normalize_trace_text("file_size = 200\nThe final answer is 200")
    tr = ArithTrace.parse(fixed)
    ok &= _check("normalized form parses and preserves the value",
                tr is not None and tr.self_consistent() and tr.denote() == 200,
                f"-> {fixed!r}")

    print("\nG2  'show the substitution' lines (three '=' signs)")
    line = "total = a + b = 3 + 4 = 7"
    raw = f"a = 3 + 0 = 3\nb = 4 + 0 = 4\n{line}\nThe final answer is 7"
    ok &= _check("original multi-'=' line is unparseable on its own",
                _STEP.match(line) is None)
    fixed = normalize_trace_text(raw)
    tr = ArithTrace.parse(fixed)
    ok &= _check("normalized trace parses and is self-consistent",
                tr is not None and tr.self_consistent() and tr.denote() == 7,
                f"-> {fixed!r}")

    print("\nG3  normalizer never changes an already-parseable line")
    good = "cost = 3 * 7 = 21"
    ok &= _check("untouched", normalize_line(good) == good)

    print("\nG4  normalizer does not invent structure from pure prose")
    prose = "First I need to figure out how many eggs are left over."
    ok &= _check("prose line is returned unchanged (still unparseable)",
                normalize_line(prose) == prose)

    print("\nG5  retroactive re-gate of the real DeepSeek-R1 raw JSON (offline)")
    raw_path = Path("results/raw/full_deepseek_deepseek-r1.json")
    if not raw_path.exists():
        print("  [SKIP] no raw DeepSeek JSON found in this checkout")
    else:
        state = json.loads(raw_path.read_text(encoding="utf-8"))
        rejected = [r for r in state["records"]
                   if r.get("status") == "C1_or_trace_consistency_rejected"]
        empty = [r for r in rejected if not (r.get("trace") or "").strip()]
        nonempty = [r for r in rejected if (r.get("trace") or "").strip()]

        before_parsed = sum(1 for r in nonempty if r.get("parsed"))
        before_consistent = sum(1 for r in nonempty if r.get("self_consistent"))
        before_buildable = sum(
            1 for r in nonempty
            if build_items_all_positions(r["problem_id"], "", r["trace"])
        )

        after_parsed = after_consistent = after_buildable = 0
        for r in nonempty:
            norm = normalize_trace_text(r["trace"])
            parsed = ArithTrace.parse(norm)
            if parsed is not None:
                after_parsed += 1
                if parsed.self_consistent():
                    after_consistent += 1
            if build_items_all_positions(r["problem_id"], "", norm):
                after_buildable += 1

        n_rej, n_empty, n_nonempty = len(rejected), len(empty), len(nonempty)
        print(f"      {n_rej} rejected records (seeds 0-2): "
             f"{n_empty} empty trace (sampling-stage; not addressed here), "
             f"{n_nonempty} non-empty text (grammar-addressable population)")
        print(f"      within the {n_nonempty} non-empty rejections:")
        print(f"        parsed (>=1 line matched):    {before_parsed:4d} -> {after_parsed:4d}")
        print(f"        self-consistent (C2):         {before_consistent:4d} -> {after_consistent:4d}")
        print(f"        buildable (full item, C1-C3): {before_buildable:4d} -> {after_buildable:4d}")
        gained = after_buildable - before_buildable
        ok &= _check(
            "normalizer recovers additional buildable items with no new model call",
            gained > 0, f"+{gained} of {n_nonempty} previously-rejected non-empty traces")
        # This is a real number computed on real, already-collected text. It is
        # NOT phi/delta/sigma: recovered items still need control/treat/placebo
        # continuations scored, which needs a live model call this environment
        # cannot make (no OPENROUTER_API_KEY). Report as gate-rate improvement
        # only, and say so.
        n_gated_all_seeds = sum(
            1 for r in state["records"] if r.get("status") == "built"
        )
        total_attempts = len(state["records"])
        old_gate = n_gated_all_seeds / total_attempts if total_attempts else float("nan")
        new_gate = (n_gated_all_seeds + after_buildable) / total_attempts if total_attempts else float("nan")
        print(f"      naive gate-rate bound if these were re-scored and all gated "
             f"(upper bound, ignores C4/self-consistency-under-edit): "
             f"{old_gate:.3f} -> {new_gate:.3f}")

        Path("results").mkdir(exist_ok=True)
        Path("results/grammar_ext_deepseek_regate.json").write_text(
            json.dumps({
                "n_rejected_seeds_0_2": n_rej,
                "n_empty_trace_not_addressed": n_empty,
                "n_nonempty_rejected": n_nonempty,
                "before": {"parsed": before_parsed, "self_consistent": before_consistent,
                          "buildable_c1_c3": before_buildable},
                "after_normalization": {"parsed": after_parsed, "self_consistent": after_consistent,
                                        "buildable_c1_c3": after_buildable},
                "note": (
                    "Offline retroactive re-gate on already-collected DeepSeek-R1 "
                    "text using denote_grammar_ext.normalize_trace_text + the "
                    "unmodified ArithTrace parser. No new model call was made. "
                    "'buildable_c1_c3' counts items for which a scoreable "
                    "treatment edit exists; it is NOT phi/delta/sigma, which "
                    "still requires live continuation calls on these normalized "
                    "traces (blocked in this environment: no OPENROUTER_API_KEY)."
                ),
            }, indent=2),
            encoding="utf-8",
        )

    print("\n" + "=" * 66)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    print("G1-G4 verify the normalizer against planted structure.")
    print("G5 is a real, model-free re-gate of already-collected DeepSeek-R1")
    print("text. It is not a new phi/delta/sigma measurement.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(self_test())
