#!/usr/bin/env python3
"""Local (Hugging Face, on-GPU) backend for DENOTE's ModelInterface.

Sibling to denote_model.py's OpenRouterModel, same contract, same prompts
(imported, not duplicated), so a local model plugs into
denote_experiments.collect_interventional() unchanged. Written for the
train-against-sensitivity demonstration (denote_tdsc.tex ~line 425): the
"before" and "after" (fine-tuned) passes both go through this class, so the
only thing that differs between them is which weights are loaded.
"""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from denote_build import ModelInterface
from denote_model import CONTINUE_SYSTEM, DIRECT_SYSTEM, TRACE_SYSTEM


class LocalHFModel(ModelInterface):
    """Runs a small instruct model locally via transformers.generate().

    Pass ``adapter_path`` to load a LoRA adapter on top of the base weights
    (the "after" pass); omit it for the base model (the "before" pass).
    Pass ``load_in_4bit=True`` for models too large to fit in fp16 on this
    machine's 6GB card (e.g. 7B+) -- uses bitsandbytes NF4 quantization.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        *,
        adapter_path: str | None = None,
        device: str = "cuda",
        max_trace_tokens: int = 700,
        max_answer_tokens: int = 80,
        trace_temperature: float = 0.7,
        load_in_4bit: bool = False,
    ):
        self.model_id = model_id
        self.max_trace_tokens = max_trace_tokens
        self.max_answer_tokens = max_answer_tokens
        self.trace_temperature = trace_temperature

        # NOTE: `self.model` must stay a *string* (the model identifier) --
        # denote_experiments.collect_interventional()/run_full() read
        # `model.model` expecting the name (for the output JSON's "model"
        # field and log lines), matching OpenRouterModel's convention. The
        # loaded weights live in `self._hf_model` instead.
        self.model = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb_config, device_map={"": 0}
            )
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16
            ).to(device)
        if adapter_path:
            hf_model = PeftModel.from_pretrained(hf_model, adapter_path)
            if not load_in_4bit:
                hf_model = hf_model.to(device)
        self._hf_model = hf_model
        self.device = device

    # --- Public ModelInterface methods ------------------------------------

    def sample_trace(self, question: str, seed: int) -> str:
        return self._generate(
            system=TRACE_SYSTEM,
            user=f"Question:\n{question}",
            do_sample=True,
            temperature=self.trace_temperature,
            seed=seed,
            max_new_tokens=self.max_trace_tokens,
        )

    def continue_from(self, question: str, trace: str, implied=None) -> str:
        del implied
        return self._generate(
            system=CONTINUE_SYSTEM,
            user=f"Question:\n{question}\n\nReasoning trace:\n{trace}",
            do_sample=False,
            temperature=0.0,
            seed=0,
            max_new_tokens=self.max_answer_tokens,
        )

    def answer_direct(self, question: str) -> str:
        return self._generate(
            system=DIRECT_SYSTEM,
            user=question,
            do_sample=False,
            temperature=0.0,
            seed=0,
            max_new_tokens=self.max_answer_tokens,
        )

    # --- Generation ---------------------------------------------------------

    def _generate(
        self,
        *,
        system: str,
        user: str,
        do_sample: bool,
        temperature: float,
        seed: int,
        max_new_tokens: int,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if do_sample:
            torch.manual_seed(seed)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self._hf_model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
