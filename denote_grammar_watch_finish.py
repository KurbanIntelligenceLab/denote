#!/usr/bin/env python3
"""Watch DeepSeek grammar re-collect; on completed_at analyze + light tex note.

Does not overwrite Table 1. Resumes the collect job if the process dies.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results" / "raw" / "full_deepseek_deepseek-r1_grammar.json"
LOG = ROOT / "results" / "logs" / "deepseek_grammar_resume.log"
ERR = ROOT / "results" / "logs" / "deepseek_grammar_resume.err"
SUMMARY = ROOT / "results" / "grammar_summary.json"
PY = os.environ.get("DENOTE_PYTHON", str(Path.home() / "anaconda3" / "python.exe"))
POLL_S = int(os.environ.get("GRAMMAR_WATCH_POLL_S", "120"))


def _completed() -> dict | None:
    if not RAW.exists():
        return None
    state = json.loads(RAW.read_text(encoding="utf-8"))
    if state.get("completed_at"):
        return state
    return None


def _denote_pids() -> list[int]:
    try:
        import psutil  # type: ignore
    except ImportError:
        # Fallback: tasklist scan via powershell
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Where-Object { $_.CommandLine -match 'output-tag.?grammar|normalize-grammar' } | "
                "Select-Object -ExpandProperty ProcessId",
            ],
            text=True,
            cwd=str(ROOT),
        )
        return [int(x) for x in out.split() if x.strip().isdigit()]
    pids = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
        except (psutil.Error, TypeError):
            continue
        if "denote_experiments.py" in cmd and "grammar" in cmd:
            pids.append(int(proc.info["pid"]))
    return pids


def _resume() -> None:
    print("[watch] resuming grammar collect", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    args = [
        PY, "-u", "denote_experiments.py", "full",
        "--models", "deepseek/deepseek-r1",
        "--limit", "150", "--seeds", "0",
        "--normalize-grammar", "--output-tag", "grammar",
        "--skip-baselines", "--skip-on-policy", "--skip-analyze",
    ]
    with LOG.open("a", encoding="utf-8") as out, ERR.open("a", encoding="utf-8") as err:
        subprocess.Popen(args, cwd=str(ROOT), stdout=out, stderr=err)


def _analyze(state: dict) -> dict:
    sys.path.insert(0, str(ROOT))
    from denote_experiments import analyze_state, _clean_floats

    row = analyze_state(state)
    row["grammar_normalized"] = state.get("grammar_normalized")
    row["max_trace_tokens"] = state.get("max_trace_tokens")
    row["output_tag"] = "grammar"
    row["source_raw"] = str(RAW.as_posix())
    SUMMARY.write_text(
        json.dumps(_clean_floats(row), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return row


def _patch_tex(row: dict) -> None:
    tex = ROOT / "denote_main.tex"
    text = tex.read_text(encoding="utf-8")
    phi = row.get("phi")
    delta = row.get("delta")
    sigma = row.get("sigma")
    n_gated = row.get("n_gated")
    gate = row.get("gate_rate")
    note = (
        f"Grammar-normalized DeepSeek-R1 refresh "
        f"(separate artifact \\texttt{{results/grammar\\_summary.json}}, "
        f"not Table~\\ref{{tab:headtohead}}): "
        f"$n_{{\\mathrm{{gated}}}}={n_gated}$, gate rate ${gate:.3f}$"
        if isinstance(gate, (int, float))
        else f"Grammar-normalized DeepSeek-R1 refresh "
        f"(\\texttt{{results/grammar\\_summary.json}})"
    )
    if isinstance(phi, (int, float)):
        note += f", $\\varphi={phi:.3f}$"
    if isinstance(delta, (int, float)):
        note += f", $\\delta={delta:.3f}$"
    if isinstance(sigma, (int, float)):
        note += f", $\\sigma={sigma:.3f}$"
    note += "."

    # Prefer replacing the Scope grammar \\rx if present
    marker = r"\rx{re-collect DeepSeek-R1 with the normalizer and a larger token budget"
    idx = text.find(marker)
    if idx >= 0:
        end = text.find("}", idx)
        # find matching close of \rx{...} — simple: first } after, but nested?
        # Scope rx is long; find closing "} Nothing in"
        end2 = text.find("}", idx)
        # walk for end of \rx{...}
        depth = 0
        j = idx
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end2 = j
                    break
            j += 1
        old = text[idx : end2 + 1]
        text = text.replace(old, note, 1)
        tex.write_text(text, encoding="utf-8")
        print("[watch] patched Scope rx with grammar numbers", flush=True)
    else:
        print("[watch] Scope rx already cleared or not found; summary written only", flush=True)


def main() -> int:
    print(f"[watch] polling every {POLL_S}s for {RAW.name}", flush=True)
    while True:
        state = _completed()
        if state is not None:
            print(f"[watch] completed_at={state['completed_at']}", flush=True)
            row = _analyze(state)
            print(json.dumps({k: row.get(k) for k in ("n_raw", "n_gated", "phi", "delta", "sigma")}, indent=2), flush=True)
            _patch_tex(row)
            subprocess.call([PY, "denote_time_log.py", "stop", "deepseek-grammar-recollect",
                             "--outcome", f"grammar done; n_gated={row.get('n_gated')}"], cwd=str(ROOT))
            # progress note
            prog = ROOT / "V2_PROGRESS.md"
            ptxt = prog.read_text(encoding="utf-8")
            ptxt = ptxt.replace(
                "`in_progress` (~38/150)",
                "`done`",
            )
            # also broader replace
            import re
            ptxt = re.sub(
                r"\| torun-reasoning-grammar-live \|.*?\| `in_progress`[^|]*\|",
                "| torun-reasoning-grammar-live | Re-collect DeepSeek-R1 with normalizer + larger token budget | `done` |",
                ptxt,
                count=1,
            )
            prog.write_text(ptxt, encoding="utf-8")
            print("[watch] done", flush=True)
            return 0
        pids = _denote_pids()
        if not pids:
            print("[watch] no grammar process; resuming", flush=True)
            _resume()
        else:
            n = 0
            if RAW.exists():
                n = len(json.loads(RAW.read_text(encoding="utf-8")).get("records", []))
            print(f"[watch] alive pids={pids} n_records={n}", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
