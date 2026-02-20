import json
import os
import re
from typing import Optional

from datasets import load_dataset
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from ..data_registry import EVAL_DATASETS

OPTION_LETTERS = ("A", "B", "C", "D")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _context_from_example(ex) -> str:
    ctx = (ex.get("ctx") or "").strip()
    if ctx:
        return ctx
    ctx_a = (ex.get("ctx_a") or "").strip()
    ctx_b = (ex.get("ctx_b") or "").strip()
    return " ".join(part for part in (ctx_a, ctx_b) if part).strip()


def _label_to_index(label) -> int:
    if isinstance(label, int):
        return label
    if label is None:
        return -1
    s = str(label).strip()
    if s.isdigit():
        idx = int(s)
        return idx if 0 <= idx < 4 else -1
    s = s.upper()
    if s in OPTION_LETTERS:
        return OPTION_LETTERS.index(s)
    return -1


def _extract_pred_index(pred_text: str, endings) -> Optional[int]:
    if not pred_text:
        return None

    lines = [ln.strip() for ln in pred_text.splitlines() if ln.strip()]
    first_line = lines[0] if lines else pred_text.strip()

    def _line_to_choice(line: str) -> Optional[int]:
        if not line:
            return None
        line = line.strip()
        line = re.sub(r"(?i)^answer\s*[:\-]?\s*", "", line).strip()

        # Direct single-token answers.
        if re.fullmatch(r"[ABCD]", line, flags=re.IGNORECASE):
            return OPTION_LETTERS.index(line.upper())
        if re.fullmatch(r"[0-3]", line):
            return int(line)

        # Option-style prefixes, e.g. "B) ...", "2. ...".
        m = re.match(r"^([ABCD])[\)\].:\-]\s*", line, flags=re.IGNORECASE)
        if m:
            return OPTION_LETTERS.index(m.group(1).upper())
        m = re.match(r"^([0-3])[\)\].:\-]\s*", line)
        if m:
            return int(m.group(1))
        return None

    def _match_ending_from_text(text: str) -> Optional[int]:
        t_norm = _normalize_text(text)
        if not t_norm:
            return None
        best_idx = None
        best_len = -1
        for i, ending in enumerate(endings):
            e_norm = _normalize_text(ending)
            if not e_norm:
                continue
            if e_norm == t_norm or e_norm in t_norm or t_norm in e_norm:
                if len(e_norm) > best_len:
                    best_idx = i
                    best_len = len(e_norm)
        return best_idx

    # Preferred: parse the first emitted line as answer.
    idx = _line_to_choice(first_line)
    if idx is not None:
        return idx

    # If the model outputs the ending text directly on the first line, use it.
    idx = _match_ending_from_text(first_line)
    if idx is not None:
        return idx

    # Secondary: explicit "Answer: X" anywhere.
    answer_match = re.search(r"(?i)\banswer\s*[:\-]?\s*([ABCD]|[0-3])\b", pred_text)
    if answer_match:
        token = answer_match.group(1).upper()
        if token in OPTION_LETTERS:
            return OPTION_LETTERS.index(token)
        return int(token)

    # Tertiary: direct letter token in full output.
    letter = re.search(r"\b([ABCD])\b", pred_text, flags=re.IGNORECASE)
    if letter:
        return OPTION_LETTERS.index(letter.group(1).upper())

    # Numeric fallback only when it appears as a standalone answer-like line.
    for line in lines[:3]:
        idx = _line_to_choice(line)
        if idx is not None:
            return idx

    # Fallback: model emitted ending text instead of option id.
    pred_norm = _normalize_text(pred_text)
    best_idx = None
    best_pos = None
    best_len = -1
    for idx, ending in enumerate(endings):
        ending_norm = _normalize_text(ending)
        if not ending_norm:
            continue
        pos = pred_norm.find(ending_norm)
        if pos == -1:
            continue
        # Prefer earliest match (often the actual answer line), then longer text.
        if best_pos is None or pos < best_pos or (pos == best_pos and len(ending_norm) > best_len):
            best_idx = idx
            best_pos = pos
            best_len = len(ending_norm)
    return best_idx


def _build_prompt(context: str, endings) -> str:
    return (
        f"Context: {context}\n\n"
        "Choose the most plausible ending.\n"
        f"A) {endings[0]}\n"
        f"B) {endings[1]}\n"
        f"C) {endings[2]}\n"
        f"D) {endings[3]}\n"
        "Answer with a single letter (A, B, C, or D).\n"
        "Answer:"
    )


def add_args(parser):
    return


def load_data(args):
    print("Loading HellaSwag...")
    ds_cfg = EVAL_DATASETS["hellaswag"]
    ds = load_dataset(ds_cfg["path"], ds_cfg["name"], split=ds_cfg["split"])
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))

    prompts = []
    golds = []
    for ex in ds:
        endings = ex.get("endings", [])
        if len(endings) != 4:
            raise ValueError(f"HellaSwag example has {len(endings)} endings (expected 4).")
        context = _context_from_example(ex)
        prompts.append(_build_prompt(context, endings))
        golds.append(_label_to_index(ex.get("label")))
    return ds, prompts, golds


def score_and_save(args, ds, prompts, golds, outputs):
    if args.wandb and wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"eval/{args.run_name}",
            config=vars(args),
        )
        wb_table = wandb.Table(
            columns=["context", "gold_idx", "pred_idx", "pred_text", "ending_a", "ending_b", "ending_c", "ending_d"]
        )

    correct = 0
    wrong_logged = 0

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{args.run_name}.jsonl")

    print("Scoring results...")
    with open(out_file, "w", encoding="utf-8") as f:
        for ex, gold, pred in tqdm(zip(ds, golds, outputs), total=len(prompts), desc="Scoring"):
            endings = ex["endings"]
            pred_idx = _extract_pred_index(pred, endings)
            is_correct = (gold >= 0) and (pred_idx is not None) and (pred_idx == gold)
            if is_correct:
                correct += 1

            res = {
                "context": _context_from_example(ex),
                "endings": endings,
                "gold_idx": gold,
                "pred_idx": pred_idx,
                "pred_text": pred,
                "correct": is_correct,
            }
            f.write(json.dumps(res) + "\n")

            if args.wandb and wandb and (not is_correct) and wrong_logged < 100:
                wb_table.add_data(
                    res["context"],
                    gold,
                    pred_idx,
                    pred,
                    endings[0],
                    endings[1],
                    endings[2],
                    endings[3],
                )
                wrong_logged += 1

    acc = correct / len(prompts) if prompts else 0.0
    print(f"\nFinal Accuracy: {acc:.2%}")

    summary = {
        "run_name": args.run_name,
        "model_name": args.model,
        "adapter": args.adapter,
        "dataset": "hellaswag",
        "split": EVAL_DATASETS["hellaswag"]["split"],
        "n": len(prompts),
        "bs": args.bs,
        "max_new_tokens": args.max_new_tokens,
        "correct": correct,
        "total": len(prompts),
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
                "eval/total": len(prompts),
                "eval/wrong_examples": wb_table,
            }
        )
        wandb.finish()
