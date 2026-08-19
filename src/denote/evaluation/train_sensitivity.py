#!/usr/bin/env python3
"""Train-against-sensitivity: DPO fine-tuning step (denote_tdsc.tex ~line 425).

Builds preference pairs from a "before" collection's genuinely derailed
records -- chosen = the trace's own clean answer, rejected = the actual
observed derailed continuation -- and trains a small LoRA adapter with
trl's DPOTrainer so the "after" pass (denote_local_collect.py --adapter ...)
can be compared against the "before" pass.

Usage:
  python denote_train_sensitivity.py \
      --before results/raw/full_qwen_qwen2.5-7b-instruct_local_before.json \
      --model-id Qwen/Qwen2.5-7B-Instruct --load-4bit \
      --output results/lora_sensitivity_qwen7b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# datasets must import before torch on this machine: importing torch first
# and then datasets (which pulls in pyarrow) segfaults on load, confirmed by
# bisecting the import order directly. Root cause not pinned down further
# (looks like a native-extension init-order conflict); the reorder is a
# reliable, harmless workaround.
from datasets import Dataset

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from denote_model import CONTINUE_SYSTEM
from denote_tdsc_verify import close


def build_pairs(before_path: str) -> list[dict]:
    data = json.loads(Path(before_path).read_text(encoding="utf-8"))
    records = data["records"]

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

    pairs = []
    for r in gated:
        at = (r.get("answers") or {}).get("treat")
        if at is None:
            continue
        if close(at, r["den_treat"]) or close(at, r["den_clean"]):
            continue  # follow or restore -- not a derailment
        rejected = (r.get("outputs") or {}).get("treat")
        if not rejected:
            continue
        prompt = (
            f"Question:\n{r['question']}\n\nReasoning trace:\n{r['trace']}"
        )
        chosen = f"The final answer is {r['den_clean']:g}"
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    pairs = build_pairs(args.before)
    print(f"[train-sensitivity] {len(pairs)} DPO pairs from {args.before}", flush=True)
    if not pairs:
        raise SystemExit("no derailed records found -- nothing to train against")
    for p in pairs:
        print(f"  chosen={p['chosen']!r} rejected={p['rejected']!r}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, quantization_config=bnb_config, device_map={"": 0}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, dtype=torch.float16
        ).to("cuda")

    # prompt/chosen/rejected need the chat template + system prompt applied,
    # matching LocalHFModel.continue_from's exact format so the adapter sees
    # the same distribution at train and eval time.
    def format_row(row):
        messages = [
            {"role": "system", "content": CONTINUE_SYSTEM},
            {"role": "user", "content": row["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt_text, "chosen": row["chosen"], "rejected": row["rejected"]}

    dataset = Dataset.from_list([format_row(p) for p in pairs])

    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    training_args = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        beta=0.1,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    result = trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    log_path = Path(args.output) / "train_log.json"
    log_path.write_text(json.dumps({
        "n_pairs": len(pairs),
        "model_id": args.model_id,
        "epochs": args.epochs,
        "final_loss": result.training_loss,
    }, indent=2), encoding="utf-8")
    print(f"[train-sensitivity] done -> {args.output} (final_loss={result.training_loss:.4f})", flush=True)


if __name__ == "__main__":
    main()
