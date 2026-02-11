import os
import json
import re
from typing import Optional

from datasets import load_dataset
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from ..data_registry import EVAL_DATASETS
from ..utils_eval import PLAIN_TEMPLATE, build_prompt


# Robust Extraction & Normalization (GSM8K-specific)
def normalize_number(s: str) -> str:
    if not s:
        return s
    s = s.replace(",", "")
    try:
        f_val = float(s)
        # If it's effectively an integer (42.0), return '42'
        if f_val.is_integer():
            return str(int(f_val))
        return str(f_val)
    except ValueError:
        return s


def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None

    # 1. Try GSM8K standard '####'
    # We take the FIRST match to avoid the "repetition loop" truncation bug.
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text.replace(",", ""))
    if match:
        return normalize_number(match.group(1).strip())

    # 2. Fallback: Last number
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if numbers:
        return normalize_number(numbers[-1].strip())

    return None


def add_args(parser):
    # GSM8K uses only the shared args for now.
    return


def load_data(args):
    # A. Data
    print("Loading GSM8K...")
    ds_cfg = EVAL_DATASETS["gsm8k"]
    ds = load_dataset(ds_cfg["path"], ds_cfg["name"], split=ds_cfg["split"])
    if args.n:
        ds = ds.select(range(args.n))

    prompts = [build_prompt(q, PLAIN_TEMPLATE) for q in ds["question"]]
    golds = ds["answer"]
    return ds, prompts, golds


def score_and_save(args, ds, prompts, golds, outputs):
    # D. W&B Setup
    if args.wandb and wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"eval/{args.run_name}",
            config=vars(args)
        )
        wb_table = wandb.Table(columns=["question", "gold_answer", "pred_answer", "pred_text"])

    # E. Score & Save
    correct = 0
    wrong_logged = 0
    results = []

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{args.run_name}.jsonl")

    print("Scoring results...")
    with open(out_file, "w") as f:
        # Use tqdm for scoring progress too, it's fast but good to see
        for q, gold, pred in tqdm(zip(ds["question"], golds, outputs), total=len(prompts), desc="Scoring"):
            # Extract & Normalize
            gold_ans = extract_answer(gold)
            pred_ans = extract_answer(pred)

            # Comparison (String equality after normalization)
            is_correct = (gold_ans is not None) and (pred_ans is not None) and (gold_ans == pred_ans)
            if is_correct:
                correct += 1

            res = {
                "question": q,
                "gold_text": gold,
                "pred_text": pred,
                "gold_ans": gold_ans,
                "pred_ans": pred_ans,
                "correct": is_correct
            }
            f.write(json.dumps(res) + "\n")
            results.append(res)

            # W&B Logging (Only wrong answers, first 100)
            if args.wandb and wandb and (not is_correct) and wrong_logged < 100:
                wb_table.add_data(q, gold_ans, pred_ans, pred)
                wrong_logged += 1

    acc = correct / len(ds)
    print(f"\nFinal Accuracy: {acc:.2%}")

    summary = {
        "run_name": args.run_name,
        "model_name": args.model,
        "adapter": args.adapter,
        "dataset": "gsm8k",
        "split": "test",
        "n": len(ds),
        "bs": args.bs,
        "max_new_tokens": args.max_new_tokens,
        "prompt_template": PLAIN_TEMPLATE,
        "correct": correct,
        "total": len(ds),
        "acc": acc,
        "predictions_jsonl": out_file,
    }

    summary_path = os.path.join(args.output_dir, f"{args.run_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, indent=2)

    print("[IO] saved:", summary_path)

    if args.wandb and wandb:
        wandb.save(summary_path)
        wandb.save(out_file)

    if args.wandb and wandb:
        wandb.log({
            "eval/acc": acc,
            "eval/correct": correct,
            "eval/total": len(ds),
            "eval/wrong_examples": wb_table
        })
        wandb.finish()
