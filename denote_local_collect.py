#!/usr/bin/env python3
"""Local-model collection for the train-against-sensitivity demonstration
(denote_tdsc.tex ~line 425). Same construction/scoring pipeline as the
OpenRouter panel, just pointed at a local Hugging Face model instead.

Reuses denote_experiments.load_arithmetic_problems() and
collect_interventional() directly, bypassing run_full()'s pilot-gate and
OpenRouter-specific model instantiation -- both exist to protect the
headline panel's integrity, which this supplementary local run does not
touch.

Usage:
  python denote_local_collect.py --limit 1319 --seeds 0 --output-tag local_before
  python denote_local_collect.py --limit 1319 --seeds 0 --output-tag local_after \
      --adapter results/lora_sensitivity_qwen0.5b
"""

from __future__ import annotations

import argparse
from pathlib import Path

from denote_experiments import collect_interventional, load_arithmetic_problems
from denote_local_model import LocalHFModel

RAW_DIR = Path("results/raw")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=1319)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path (after-pass)")
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--load-4bit", action="store_true",
                         help="bitsandbytes NF4 quantization, for 7B+ models on 6GB VRAM")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    problems = load_arithmetic_problems(args.limit, include_math=True)
    print(f"[local-collect] {len(problems)} problems, seeds={seeds}, "
          f"adapter={args.adapter or '(none, base weights)'}, "
          f"4bit={args.load_4bit}", flush=True)

    model = LocalHFModel(args.model_id, adapter_path=args.adapter, load_in_4bit=args.load_4bit)
    slug = args.model_id.lower().replace("/", "_")
    output_path = RAW_DIR / f"full_{slug}_{args.output_tag}.json"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    state = collect_interventional(
        model, problems, seeds=seeds, output_path=output_path,
        include_baselines=args.include_baselines,
    )
    n_records = len(state.get("records", []))
    print(f"[local-collect] done -> {output_path} ({n_records} records)", flush=True)


if __name__ == "__main__":
    main()
