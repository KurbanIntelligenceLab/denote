# DENOTE

Code and data for the corruption-test faithfulness analysis in *What
Chain-of-Thought Corruption Scores Identify: Bounds, Ranking Uncertainty,
and Audits* (IEEE TDSC, special issue on Trust, Privacy, and Security in
Intelligent and Agentic Systems).

This repository contains the construction, evaluator, metric, and analysis
code, together with the sampled traces and raw per-model records the
paper's reported numbers are computed from. Datasets: the `gsm8k` main
test split and the `competition_math` numeric-answer test subset (MIT
licence for GSM8K; MATH licence per its dataset card). Both are public;
neither is redistributed here.

## Setup

```bash
pip install -r requirements.txt   # openai, tenacity, python-dotenv, sympy
pip install -e .                  # registers the denote package (src/denote/)
cp .env.example .env              # then fill in OPENROUTER_API_KEY
```

Live experiment scripts call the OpenRouter API and need a key at
[openrouter.ai](https://openrouter.ai). Analysis scripts that only read
`results/` do not need one.

## Layout

```
src/denote/
├── construction/   # build treat/control/placebo items from source problems
├── evaluation/     # OpenRouter + local (HF) model clients, run orchestration,
│                   # baseline/legacy scoring, judge comparator, seed replication
├── metrics/        # σ/φ/δ classification per item
└── analysis/       # gating/aggregation/bootstrap, CERTIFY, proposition checks,
                    # parser and extraction audits, ranking geometry

results/
├── raw/            # raw per-model API records
├── summaries/      # derived σ/φ/δ panels, pilot and audit summaries
├── audits/         # human-in-the-loop audit sheets and scored results
└── tables/         # generated LaTeX source for the paper's Table I
```

Each module is documented at the top of its file; start from
`denote.analysis.tdsc_verify` and `denote.analysis.experiments` to see how the
pieces fit together.

## Reproducing the analysis

No API calls required — everything below reads `results/` as shipped, run
from the repo root after `pip install -e .`:

```bash
python -m denote.analysis.tdsc_verify --all
python -m denote.analysis.experiments analyze --bootstrap 10000
python -m denote.analysis.tdsc_certify
python -m denote.analysis.tdsc_verify_theory
python -m denote.analysis.code_experiments analyze   # DENOTE-Code panel (supp. Appendix F)
```

`denote.analysis.tdsc_verify --all` recomputes σ/φ/δ for the six-model
headline panel (Table I) from the raw records, checks the paper's printed
table against that recomputation, and runs the ranking/grid/audit/certify
checks described in the paper. It does not cover the seed-replication
table, the baseline/metric-comparison panel, or the eight-model ranking
extension — those are checked by hand against `results/raw/` and
`results/summaries/` rather than by an automated script.
`denote.analysis.code_experiments analyze` independently recomputes the
DENOTE-Code panel (supplement Appendix F) from `results/raw/code_full_*.json`.
`denote.analysis.tdsc_verify_theory` is a pure math/combinatorics check of
the paper's propositions; it reads no experimental data and asserts no
empirical claim.

## License

Code: MIT. Data: GSM8K (MIT) and MATH (per its own dataset card) source
problems are not redistributed; derived traces and records are released
for research use.
