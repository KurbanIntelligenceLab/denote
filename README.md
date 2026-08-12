# DENOTE

Code and data for the corruption-test faithfulness analysis in *DENOTE:
Corruption Scores Bound Chain-of-Thought Faithfulness From Above* (IEEE
TDSC, special issue on Safety, Alignment, and Responsibility of Large
Language Models).

This repository contains the construction, evaluator, metric, and analysis
code, together with the sampled traces and raw per-model records the
paper's reported numbers are computed from. Datasets: the `gsm8k` main
test split and the `competition_math` numeric-answer test subset (MIT
licence for GSM8K; MATH licence per its dataset card). Both are public;
neither is redistributed here.

## Setup

```
pip install -r requirements.txt   # openai, tenacity, python-dotenv, sympy
cp .env.example .env              # then fill in OPENROUTER_API_KEY
```

Live experiment scripts call the OpenRouter API and need a key at
[openrouter.ai](https://openrouter.ai). Analysis scripts that only read
`results/` do not need one.

## Layout

| Path | What it is |
|---|---|
| `denote_build.py`, `denote_code_build.py` | Construction: build treat/control/placebo items from source problems |
| `denote_model.py`, `denote_run.py` | Evaluator: OpenRouter client and run orchestration |
| `denote_metrics.py` | Metric: σ/φ/δ classification per item |
| `denote_experiments.py` | Primary analysis pipeline (gating, aggregation, bootstrap) |
| `denote_tdsc_verify.py` | Independent recomputation of every quantity the paper reports; `--all` runs every mode |
| `denote_tdsc_certify.py` | CERTIFY procedure (audit-based certification) |
| `denote_tdsc_verify_theory.py` | Line-by-line verification of the paper's numbered propositions |
| `denote_legacy_score.py`, `denote_seed_replicate.py`, `denote_rerun_treat.py`, `denote_grammar_ext.py`, `denote_grammar_watch_finish.py`, `denote_h4_e0_auc.py`, `denote_logit_margin_probe.py`, `denote_logprob_probe.py`, `denote_baselines.py` | Evaluator variants: the legacy arm, seed replication, treat-arm reruns, grammar-normalized collection, the labelled-E0 harness, and the baseline-metric panel |
| `denote_v1_parser_audit.py`, `denote_v2_audit.py`, `denote_grammar_analyze.py`, `denote_h4_common_pop.py`, `denote_code_experiments.py`, `compute_legacy_summary.py` | Analysis: parser audits, derailment-vs-extraction adjudication, grammar-panel summaries, common-population wiring, the Denote-Code (MBPP) panel |
| `results/raw/` | Raw per-model API records (one file per model; `_grammar` suffix marks the grammar-normalized re-collection) |
| `results/*.json`, `results/*.csv`, `results/tables.tex` | Derived summaries, audit sheets, and generated table source the paper's numbers trace back to |

## Reproducing the analysis

No API calls required — everything below reads `results/` as shipped:

```
python denote_tdsc_verify.py --all
python denote_experiments.py analyze --bootstrap 10000
python denote_tdsc_certify.py
python denote_tdsc_verify_theory.py
```

`denote_tdsc_verify.py --all` recomputes every reported σ/φ/δ from the raw
records, checks the paper's printed table against that recomputation, and
runs the ranking/grid/audit/certify checks described in the paper.

## License

Code: MIT. Data: GSM8K (MIT) and MATH (per its own dataset card) source
problems are not redistributed; derived traces and records are released
for research use.
