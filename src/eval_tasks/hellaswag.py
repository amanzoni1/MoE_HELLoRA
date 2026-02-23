import json
import os
from typing import List

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

try:
    import wandb
except ImportError:
    wandb = None

from ..config import TRAIN_CFG
from ..data_registry import EVAL_DATASETS

OPTION_IDS = ("0", "1", "2", "3")
CUSTOM_EVAL = True


def _context_from_example(ex) -> str:
    ctx = (ex.get("ctx") or "").strip()
    if ctx:
        return ctx
    ctx_a = (ex.get("ctx_a") or "").strip()
    ctx_b = (ex.get("ctx_b") or "").strip()
    return " ".join(part for part in (ctx_a, ctx_b) if part).strip()


def _label_to_index(label) -> int:
    if isinstance(label, int):
        return label if 0 <= label < 4 else -1
    if label is None:
        return -1
    s = str(label).strip()
    if s.isdigit():
        idx = int(s)
        return idx if 0 <= idx < 4 else -1
    return -1


def _build_prefix(context: str, endings: List[str]) -> str:
    # Keep eval prefix aligned with training formatting.
    options = "\n".join(f"  {i}) {e}" for i, e in enumerate(endings))
    return f"{context}\n{options}\nAnswer: "


def add_args(parser):
    parser.add_argument(
        "--hs_length_norm",
        action="store_true",
        help="Length-normalize option log-likelihood by number of option tokens.",
    )
    parser.add_argument(
        "--hs_max_len",
        type=int,
        default=TRAIN_CFG.max_len,
        help=f"Max sequence length for HellaSwag scoring (default: {TRAIN_CFG.max_len}).",
    )
    parser.add_argument(
        "--hs_options_per_pass",
        type=int,
        default=1,
        choices=[1, 2, 4],
        help="How many options to score in one forward pass. Higher can be faster but uses more VRAM.",
    )


def _score_option_batch(
    model,
    tokenizer,
    prefix_texts: List[str],
    option_texts: List[str],
    max_len: int,
    length_norm: bool,
) -> List[float]:
    full_texts = [f"{p}{opt.strip()}" for p, opt in zip(prefix_texts, option_texts)]

    tok_full = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )
    tok_prefix = tokenizer(
        prefix_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )

    device = model.device
    input_ids = tok_full["input_ids"].to(device)
    attention_mask = tok_full["attention_mask"].to(device)
    prefix_lens = tok_prefix["attention_mask"].sum(dim=1).to(device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].bool()

    token_logp = torch.log_softmax(shift_logits, dim=-1).gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_positions = torch.arange(1, input_ids.shape[1], device=device).unsqueeze(0)
    # Boundary of continuation tokens in the full sequence.
    # Works for both right- and left-padding (left_pad=0 for right-padding).
    left_pad = attention_mask.to(torch.int64).argmax(dim=1)
    prefix_end_positions = left_pad + prefix_lens
    candidate_mask = token_positions >= prefix_end_positions.unsqueeze(1)
    valid_mask = shift_mask & candidate_mask

    token_counts = valid_mask.sum(dim=1)
    seq_scores = (token_logp * valid_mask).sum(dim=1)
    if length_norm:
        seq_scores = seq_scores / token_counts.clamp(min=1)
    # If truncation removed the whole option, make this candidate impossible.
    seq_scores = torch.where(token_counts > 0, seq_scores, torch.full_like(seq_scores, -1e9))
    return seq_scores.detach().cpu().tolist()


def run_custom_eval(args):
    if args.backend != "hf":
        print(f"[HellaSwag] log-likelihood ranking uses HF logits; ignoring backend='{args.backend}'.")

    print("Loading HellaSwag...")
    ds_cfg = EVAL_DATASETS["hellaswag"]
    ds = load_dataset(ds_cfg["path"], ds_cfg["name"], split=ds_cfg["split"])
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))

    prefixes = []
    endings_list = []
    golds = []
    for ex in ds:
        endings = ex.get("endings", [])
        if len(endings) != 4:
            raise ValueError(f"HellaSwag example has {len(endings)} endings (expected 4).")
        prefixes.append(_build_prefix(_context_from_example(ex), endings))
        endings_list.append(endings)
        golds.append(_label_to_index(ex.get("label")))

    if args.wandb and wandb:
        wandb.init(project=args.wandb_project, name=f"eval/{args.run_name}", config=vars(args))
        wb_table = wandb.Table(
            columns=["context", "gold_idx", "pred_idx", "logp_0", "logp_1", "logp_2", "logp_3", "ending_0", "ending_1", "ending_2", "ending_3"]
        )

    print(f"[HF] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if args.adapter:
        print(f"[HF] Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model.config.use_cache = False
    model.eval()

    n = len(prefixes)
    option_scores = [[0.0, 0.0, 0.0, 0.0] for _ in range(n)]
    options_per_pass = max(1, min(4, int(args.hs_options_per_pass)))
    option_groups = [list(range(i, min(i + options_per_pass, 4))) for i in range(0, 4, options_per_pass)]
    print(f"[HellaSwag] options_per_pass={options_per_pass} (effective batch = bs x {options_per_pass})")
    for group in option_groups:
        group_name = ",".join(OPTION_IDS[i] for i in group)
        desc = f"Scoring options {group_name}"
        for start in tqdm(range(0, n, args.bs), desc=desc):
            end = min(start + args.bs, n)
            batch_prefixes = prefixes[start:end]
            flat_prefixes = []
            flat_options = []
            flat_meta = []  # (global_example_idx, option_idx)
            for local_i, ex_endings in enumerate(endings_list[start:end]):
                global_i = start + local_i
                for opt_idx in group:
                    flat_prefixes.append(batch_prefixes[local_i])
                    flat_options.append(ex_endings[opt_idx])
                    flat_meta.append((global_i, opt_idx))

            flat_scores = _score_option_batch(
                model=model,
                tokenizer=tokenizer,
                prefix_texts=flat_prefixes,
                option_texts=flat_options,
                max_len=args.hs_max_len,
                length_norm=args.hs_length_norm,
            )
            for (global_i, opt_idx), score in zip(flat_meta, flat_scores):
                option_scores[global_i][opt_idx] = float(score)

    pred_idxs = [max(range(4), key=lambda i: row[i]) for row in option_scores]
    correct = sum(1 for g, p in zip(golds, pred_idxs) if g >= 0 and g == p)
    total = len(golds)
    acc = correct / total if total else 0.0

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{args.run_name}.jsonl")
    wrong_logged = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            res = {
                "context": _context_from_example(ex),
                "endings": endings_list[i],
                "gold_idx": golds[i],
                "pred_idx": pred_idxs[i],
                "pred_text": OPTION_IDS[pred_idxs[i]],
                "choice_logprobs": option_scores[i],
                "correct": bool(golds[i] >= 0 and pred_idxs[i] == golds[i]),
            }
            f.write(json.dumps(res) + "\n")
            if args.wandb and wandb and (not res["correct"]) and wrong_logged < 100:
                wb_table.add_data(
                    res["context"],
                    res["gold_idx"],
                    res["pred_idx"],
                    res["choice_logprobs"][0],
                    res["choice_logprobs"][1],
                    res["choice_logprobs"][2],
                    res["choice_logprobs"][3],
                    res["endings"][0],
                    res["endings"][1],
                    res["endings"][2],
                    res["endings"][3],
                )
                wrong_logged += 1

    print(f"\nFinal Accuracy: {acc:.2%}")
    summary = {
        "run_name": args.run_name,
        "model_name": args.model,
        "adapter": args.adapter,
        "dataset": "hellaswag",
        "split": EVAL_DATASETS["hellaswag"]["split"],
        "n": total,
        "bs": args.bs,
        "max_seq_len": args.hs_max_len,
        "options_per_pass": options_per_pass,
        "metric": "option_loglikelihood_acc",
        "length_normalized": bool(args.hs_length_norm),
        "correct": correct,
        "total": total,
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
        wandb.log(
            {
                "eval/acc": acc,
                "eval/correct": correct,
                "eval/total": total,
                "eval/wrong_examples": wb_table,
            }
        )
        wandb.finish()
