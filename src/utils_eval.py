import gc
import random
import hashlib
from typing import List

import torch
from tqdm.auto import tqdm

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None


def set_repro(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


# Model Backend: Merge
def merge_and_save(base_model: str, adapter: str, out_dir: str):
    """Merges LoRA into base model safely with sharding."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"[MERGE] Loading base: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Use bfloat16 for Ampere (A40/A6000)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )

    print(f"[MERGE] Loading adapter: {adapter}")
    model = PeftModel.from_pretrained(base, adapter)
    model = model.merge_and_unload()

    print(f"[MERGE] Saving to: {out_dir}")
    # Shard size 2GB to prevent OOM
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(out_dir)

    print("[MERGE] Cleaning up VRAM...")

    del model, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print("[MERGE] Done. GPU is pristine.")

    return out_dir


# Inference Engines
def run_inference_vllm(model_path: str, prompts: List[str], max_tokens: int, tp: int) -> List[str]:
    if LLM is None:
        raise ImportError("vLLM not installed.")

    print(f"[VLLM] Initializing engine (max_tokens={max_tokens})...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        trust_remote_code=True,
        gpu_memory_utilization=0.85
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        stop=["\nQuestion:", "\n\nQuestion:"]
    )

    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text for out in outputs]

def run_inference_hf(model_path: str, adapter: str, prompts: List[str], max_tokens: int, bs: int) -> List[str]:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"[HF] Initializing (Batch Size={bs})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if adapter:
        print(f"[HF] Loading LoRA: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()

    results: List[str] = []

    for i in tqdm(range(0, len(prompts), bs), desc="HF Inference"):
        batch_prompts = prompts[i : i + bs]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        input_seq_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_only = outputs[:, input_seq_len:]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        results.extend(decoded)

    return results
