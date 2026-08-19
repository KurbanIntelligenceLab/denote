#!/usr/bin/env python3
"""
denote_code_build.py

Construction pipeline for the DENOTE-CODE second domain (MBPP / HumanEval).
This is the code-execution analogue of denote_build.py's ArithTrace: it turns
a real function (canonical solution or, in production, a model-elicited
trace in the same shape) into gated benchmark items with a computed
denotation and matched edits, using a real interpreter as the evaluator
instead of a hand-written parser.

Design, mirroring the arithmetic domain
----------------------------------------
  "step"        -> a top-level ``name = expr`` assignment directly in the
                   function body (not nested inside a loop/if/with), so a
                   resume point has one unambiguous execution.
  "denotation"  -> [[T]]: install the captured local state at that
                   assignment and run the remaining body (code_denote), i.e.
                   exactly the evaluator already defined in denote_build.py.
  "edit"        -> perturb one captured local variable's value.
  C1            -> function has at least one qualifying resume point and the
                   sandboxed tail executes without error.
  C2            -> re-running the tail from the captured state reproduces the
                   function's real return value (self-consistency).
  C3            -> the edited denotation is not already a literal visible in
                   the tail source (is_copyable, generalised to non-numeric
                   values).
  "placebo"     -> perturbing a captured variable that does not change the
                   tail's result (found empirically, not by static analysis:
                   this generalises across arbitrary Python where a static
                   def-use graph would be its own research project).

What this module does NOT do
-----------------------------
It does not call models. Eliciting a state-assertion trace from a model
(the code analogue of ``sample_trace`` for arithmetic) needs the serving
stack; this module operates on a supplied source (in the offline self-test,
the dataset's own reference solution) so the construction machinery can be
validated without any API key. ``denote_code_experiments.py`` wires this to
OpenRouterModel for the live run.

Sandboxing: ``code_denote`` (denote_build.py) uses a restricted-builtins
``exec``, not process isolation. That is adequate for trusted benchmark
source (MBPP/HumanEval reference solutions); it is NOT a sandbox against
adversarial model output. Before pointing this at model-generated code,
wrap ``_run_tail`` in a subprocess with a wall-clock timeout and a resource
limit, per the warning already in denote_build.code_denote's docstring.

Run the self-test (offline, downloads MBPP over the network, no API key):
    python3 denote_code_build.py
"""

from __future__ import annotations

import ast
import copy
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from denote_build import ModelInterface, code_denote, is_copyable

BASE_SEED = 0
_STATE_TIMEOUT_S = 2.0  # wall-clock budget per tail execution (see _run_tail)


# ---------------------------------------------------------------------------
# Parsing a canonical solution / test call
# ---------------------------------------------------------------------------

@dataclass
class CodeCall:
    func_name: str
    args: tuple
    expected: Any


def parse_test_call(test_str: str) -> CodeCall | None:
    """Parse ``assert f(1, 2) == 3`` into a structured call via ast (no exec
    of untrusted literals; ast.literal_eval only).

    When: load_mbpp_problems(), to get one concrete (args, expected) call per
    problem for building a real execution.
    """
    try:
        tree = ast.parse(test_str.strip(), mode="exec")
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None
    test = tree.body[0].test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return None
    call = test.left
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.keywords:
        return None
    try:
        args = tuple(ast.literal_eval(a) for a in call.args)
        expected = ast.literal_eval(test.comparators[0])
    except (ValueError, SyntaxError, TypeError):
        return None
    return CodeCall(call.func.id, args, expected)


def _single_function(source: str) -> ast.FunctionDef | None:
    """The one top-level function def in a canonical solution, or None.

    When: build_code_item(), first parsing step (C1's function-shape half).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return defs[0] if len(defs) == 1 else None


def _tail_source(source: str, body: list[ast.stmt], start: int,
                 state_keys: list[str]) -> str:
    """Render body[start:] as a nested function so real Python ``return``
    semantics (early exit, including from inside a loop/if) are preserved,
    then call it and bind the result to ``__result__``.

    An earlier version rewrote ``return X`` to ``__result__ = X`` textually;
    that is wrong whenever the tail branches, because assignment does not
    exit the way ``return`` does, so a later statement can silently
    overwrite the result (caught by the self-test's branching case, K2).
    Wrapping in a real function sidesteps this entirely.

    The wrapped function's parameters default to the captured state
    (``def __tail__(x=x, y=y, ...)``): default-argument expressions are
    evaluated against the *caller's* namespace (the state dict installed as
    exec locals), while the function body then sees them as its own locals,
    which is what makes the captured state visible inside a nested function
    without relying on exec's globals dict.

    Line-slicing rather than ast.unparse (Python 3.9+) so this runs on
    Python 3.8.

    When: build_code_item(), once per candidate resume point.
    """
    lines = source.splitlines()
    first, last = body[start].lineno, body[-1].end_lineno
    raw = lines[first - 1:last]
    indents = [len(l) - len(l.lstrip()) for l in raw if l.strip()]
    strip = min(indents) if indents else 0
    dedented = [l[strip:] if len(l) >= strip else l for l in raw]
    indented = "\n".join(("    " + l if l.strip() else l) for l in dedented)
    params = ", ".join(f"{k}={k}" for k in state_keys)
    return f"def __tail__({params}):\n{indented}\n    return None\n__result__ = __tail__()\n"


def _literals_in_source(source: str) -> list[Any]:
    """Every literal constant visible in ``source`` (for the C3 copy check).

    When: build_code_item(), once per candidate edit.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool, str)):
            out.append(node.value)
    return out


def _is_copyable_generic(target: Any, literals: list[Any]) -> bool:
    """C3 for the code domain's richer value space: numeric values reuse the
    arithmetic tolerance check; everything else uses exact equality.

    When: build_code_item(), scoring the treatment/copyable edits.
    """
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        return any(target == lit for lit in literals if type(lit) is type(target))
    numeric_literals = [lit for lit in literals if isinstance(lit, (int, float)) and not isinstance(lit, bool)]
    return is_copyable(float(target), [float(v) for v in numeric_literals]) if numeric_literals else False


# ---------------------------------------------------------------------------
# Sandboxed tail execution (see module docstring: adequate for trusted
# reference source, not for adversarial model output without the subprocess
# wrapper called out above)
# ---------------------------------------------------------------------------

def _run_tail(tail_source: str, state: dict) -> Any:
    """code_denote with a wall-clock guard using SIGALRM where available,
    falling back to no guard on platforms without it (e.g. Windows) — the
    caller must not point this at adversarial source on those platforms
    without adding a process-based timeout.

    When: every candidate resume point / edit inside build_code_item().
    """
    try:
        import signal
    except ImportError:
        signal = None
    if signal is not None and hasattr(signal, "SIGALRM"):
        def _timeout(_sig, _frame):
            raise TimeoutError("tail execution exceeded budget")
        old = signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, _STATE_TIMEOUT_S)
        try:
            return code_denote(tail_source, state, 0)
        except Exception:
            return None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    # No SIGALRM (Windows): best-effort, no wall-clock guard. Fine for
    # trusted MBPP/HumanEval reference source; do not use for model output
    # without the subprocess-based sandbox called out in the module docstring.
    try:
        return code_denote(tail_source, state, 0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# State capture at a resume point
# ---------------------------------------------------------------------------

def _capture_state_before(func_name: str, module_source: str, args: tuple,
                          target_lineno: int) -> dict | None:
    """Call the function for real and snapshot its locals just before the
    statement at ``target_lineno`` executes.

    When: build_code_item(), once per candidate resume point.
    """
    ns: dict = {}
    try:
        exec(compile(module_source, "<denote-code-src>", "exec"), ns)
    except Exception:
        return None
    func = ns.get(func_name)
    if func is None:
        return None

    captured: list[dict] = []
    seen_once = {"hit": False}

    def _tracer(frame, event, _arg):
        if event == "line" and frame.f_code.co_name == func_name \
                and frame.f_lineno == target_lineno and not seen_once["hit"]:
            seen_once["hit"] = True
            captured_local = {
                k: v for k, v in frame.f_locals.items()
                if not k.startswith("__") and _is_snapshottable(v)
            }
            captured.append(copy.deepcopy(captured_local))
            return None  # stop tracing this frame; snapshot is enough
        return _tracer

    old_trace = sys.gettrace()
    sys.settrace(_tracer)
    try:
        func(*copy.deepcopy(args))
    except Exception:
        return None
    finally:
        sys.settrace(old_trace)
    return captured[0] if captured else None


def _is_snapshottable(value: Any) -> bool:
    """Restrict captured state to JSON-ish types the sandboxed re-exec can
    safely re-install (no file handles, generators, etc.)."""
    return isinstance(value, (int, float, bool, str, list, tuple, dict, type(None)))


def _perturb_candidates(value: Any) -> list[Any]:
    """Type-appropriate edit candidates for one captured local.

    When: build_code_item(), searching for a treatment/placebo edit.
    """
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [value + 3]
    if isinstance(value, float):
        return [value + 3.0]
    if isinstance(value, str) and value:
        return [value + "_x"]
    if isinstance(value, list) and value and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        new = list(value)
        new[0] = new[0] + 3
        return [new]
    return []


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class CodeItem:
    problem_id: str
    prompt: str
    func_name: str
    den_clean: Any
    conditions: dict          # name -> (state_dict, tail_source)
    treat_denotation: Any
    placebo_denotation: Any | None
    copyable_denotation: Any | None
    resume_lineno: int
    edited_var: str
    trace_length: int


def build_code_item(problem_id: str, prompt: str, source: str,
                    call: CodeCall, delta_note: str = "") -> CodeItem | None:
    """Parse -> pick a resume point -> gate C1/C2 -> derive treat/placebo/copy.

    Mirrors denote_build.build_items(): returns None on any gate failure.

    When: once per MBPP/HumanEval problem, either in the offline self-test
    (source = the dataset's reference solution) or, once wired to a model, on
    a model-elicited solution.
    """
    func_def = _single_function(source)
    if func_def is None:                                   # C1a: shape
        return None
    try:
        real_output = _run_tail(source + f"\n__result__ = {func_def.name}(*__args__)\n",
                                {"__args__": call.args})
    except Exception:
        real_output = None
    if real_output is None or real_output != call.expected:
        return None                                        # reference itself must be right

    top_assigns = [
        (i, stmt) for i, stmt in enumerate(func_def.body)
        if isinstance(stmt, ast.Assign) and i < len(func_def.body) - 1
    ]
    for idx, stmt in top_assigns:
        state = _capture_state_before(func_def.name, source, call.args, stmt.lineno)
        if not state:
            continue
        tail_src = _tail_source(source, func_def.body, idx, list(state.keys()))
        den_clean = _run_tail(tail_src, state)
        if den_clean is None or den_clean != real_output:   # C1b/C2: self-consistency
            continue

        numeric_vars = [k for k, v in state.items() if _perturb_candidates(v)]
        if not numeric_vars:
            continue

        treat_var = treat_val = treat_den = None
        for var in numeric_vars:
            for cand in _perturb_candidates(state[var]):
                ov = dict(state); ov[var] = cand
                den = _run_tail(tail_src, ov)
                if den is None or den == den_clean:
                    continue
                if _is_copyable_generic(den, _literals_in_source(tail_src)):
                    continue                                # C3
                treat_var, treat_val, treat_den = var, cand, den
                break
            if treat_var:
                break
        if treat_var is None:
            continue                                        # no non-copyable edit found

        placebo_den = placebo_state = None
        for var in numeric_vars:
            if var == treat_var:
                continue
            for cand in _perturb_candidates(state[var]):
                ov = dict(state); ov[var] = cand
                den = _run_tail(tail_src, ov)
                if den is not None and den == den_clean:
                    placebo_state = dict(state); placebo_state[var] = cand
                    placebo_den = den
                    break
            if placebo_den is not None:
                break

        copy_den = copy_state = None
        last_num_vars = [k for k in numeric_vars if isinstance(state[k], (int, float)) and not isinstance(state[k], bool)]
        if last_num_vars:
            var = last_num_vars[-1]
            literal_candidates = [v for v in _literals_in_source(tail_src)
                                  if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if literal_candidates:
                target_literal = literal_candidates[0]
                ov = dict(state); ov[var] = target_literal
                den = _run_tail(tail_src, ov)
                if den is not None:
                    copy_state, copy_den = ov, den

        conditions = {
            "control": (state, tail_src),
            "treat": ({**state, treat_var: treat_val}, tail_src),
            "placebo": (placebo_state, tail_src) if placebo_state else None,
            "copyable": (copy_state, tail_src) if copy_state else None,
        }
        return CodeItem(
            problem_id, prompt, func_def.name, den_clean, conditions,
            treat_den, placebo_den, copy_den, stmt.lineno, treat_var,
            len(func_def.body),
        )
    return None                                             # no qualifying resume point


# ---------------------------------------------------------------------------
# Answer extraction (code domain: richer value space than a bare float)
# ---------------------------------------------------------------------------

_RESULT_LINE = re.compile(r"(?:final result|result|answer)\s*(?:is|:|=)\s*(.+)$", re.I | re.M)


def extract_code_answer(text: str) -> Any:
    """Deterministic extractor for a stated Python literal result.

    Tries the last '<label> is/:/= <literal>' line via ast.literal_eval, then
    the last literal-looking token in the text. Returns None on failure.

    When: scoring control/treat/placebo/copyable continuations, same role as
    denote_build.extract_answer() for the arithmetic domain.
    """
    for m in reversed(list(_RESULT_LINE.finditer(text))):
        candidate = m.group(1).strip().rstrip(".")
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
    stripped = text.strip().splitlines()[-1].strip() if text.strip() else ""
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return None


# ---------------------------------------------------------------------------
# Dataset loading (MBPP sanitized)
# ---------------------------------------------------------------------------

def load_mbpp_problems(limit: int | None = None) -> list[dict]:
    """Public MBPP (sanitized) problems with a parsed (args, expected) call.

    Each dict: problem_id, prompt, code (reference solution), call (CodeCall).
    Rows whose first test assert does not parse into a literal call are
    skipped (this is itself the first construction gate, reported by the
    self-test below).

    When: denote_code_experiments.py pilot/full, and the offline self-test.
    """
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    problems = []
    for row in ds:
        call = None
        for test_str in row.get("test_list", []):
            call = parse_test_call(test_str)
            if call is not None:
                break
        problems.append({
            "problem_id": f"mbpp:test:{row['task_id']}",
            "prompt": row["prompt"],
            "code": row["code"],
            "call": call,
        })
        if limit is not None and len(problems) >= limit:
            break
    return problems


# ---------------------------------------------------------------------------
# ModelInterface for the code domain (mirrors denote_build.ModelInterface)
# ---------------------------------------------------------------------------

class CodeModelInterface(ModelInterface):
    """Same three-method contract as the arithmetic ModelInterface. A real
    implementation (denote_code_experiments.OpenRouterCodeModel) elicits a
    state-assertion trace instead of a bare arithmetic one; nothing else in
    this pipeline needs to change."""


# ---------------------------------------------------------------------------
# Self-test: construction validity on real MBPP source, no model calls
# ---------------------------------------------------------------------------

def _check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    return bool(cond)


def self_test() -> int:
    print("denote_code_build self-test")
    print("=" * 66)
    ok = True

    print("\nK1  test-call parsing")
    call = parse_test_call('assert add(2, 3) == 5')
    ok &= _check("simple call parses", call is not None and call.func_name == "add"
                and call.args == (2, 3) and call.expected == 5)
    ok &= _check("non-assert rejected", parse_test_call("x = 1") is None)

    print("\nK2  tail wrapping preserves branching return semantics")
    src = "def f(x):\n    y = x + 1\n    if y > 0:\n        return y\n    return -1\n"
    func = _single_function(src)
    ok &= _check("function found", func is not None)
    tail = _tail_source(src, func.body, 0, ["x"])
    ok &= _check("branch that returns early is not overwritten by later code",
                code_denote(tail, {"x": 4}, 0) == 5,
                f"code_denote={code_denote(tail, {'x': 4}, 0)}  (x=4 -> y=5>0 -> should return 5, not fall through to -1)")
    ok &= _check("other branch reaches the fallthrough return",
                code_denote(_tail_source(src, func.body, 0, ["x"]), {"x": -10}, 0) == -1,
                f"code_denote={code_denote(tail, {'x': -10}, 0)}")

    print("\nK3  hand-built item on a tiny real function")
    # total = x + 5 depends only on x; y (and the aside computed from it) is
    # dead with respect to the answer, exactly like the arithmetic domain's
    # "aside" steps -- the empirical feeds/non-feeds probe should find that.
    src2 = "def f(x, y):\n    aside = y * 2\n    total = x + 5\n    return total\n"
    it = build_code_item("t1", "add 5", src2, CodeCall("f", (3, 9), 8))
    ok &= _check("item built", it is not None)
    if it is not None:
        ok &= _check("den_clean matches real output", it.den_clean == 8, f"got {it.den_clean}")
        ok &= _check("treatment denotation differs from clean",
                    it.treat_denotation != it.den_clean, f"{it.treat_denotation} vs {it.den_clean}")
        ok &= _check("edited variable (x) is the one feeding the result",
                    it.edited_var == "x", f"edited_var={it.edited_var}")
        ok &= _check("placebo found on a variable (y) that does not feed the result",
                    it.conditions["placebo"] is not None
                    and "y" in it.conditions["placebo"][0]
                    and it.conditions["placebo"][0]["y"] != 9)

    print("\nK4  answer extraction (richer value space)")
    ok &= _check("int", extract_code_answer("The final result is 7") == 7)
    ok &= _check("string", extract_code_answer('result: "abc"') == "abc")
    ok &= _check("list", extract_code_answer("answer = [1, 2, 3]") == [1, 2, 3])
    ok &= _check("bare literal fallback", extract_code_answer("42") == 42)
    ok &= _check("unparseable is None", extract_code_answer("I am not sure") is None)

    print("\nK5  construction survival on real MBPP (sanitized), no model calls")
    try:
        problems = load_mbpp_problems()
    except Exception as exc:
        print(f"  [SKIP] could not load MBPP ({exc}); K5 needs network access")
        problems = []
    n = len(problems)
    n_call_parsed = sum(1 for p in problems if p["call"] is not None)
    built = 0
    for p in problems:
        if p["call"] is None:
            continue
        try:
            item = build_code_item(p["problem_id"], p["prompt"], p["code"], p["call"])
        except Exception:
            item = None
        if item is not None:
            built += 1
    if n:
        print(f"      n={n}  test-call parsed={n_call_parsed} "
             f"({n_call_parsed/n:.1%})  built (C1-C3)={built} ({built/n:.1%})")
        ok &= _check("at least one real MBPP problem builds end-to-end",
                    built > 0, f"{built}/{n}")
    else:
        print("      skipped (no network / dataset unavailable)")

    print("\n" + "=" * 66)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    print("K1-K4 verify the construction code against planted structure.")
    print("K5 is a real, model-free construction-survival count on public")
    print("MBPP source; it is NOT a measurement of any model's behaviour.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(self_test())
