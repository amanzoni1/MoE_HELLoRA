import json
import os
import re
import subprocess
from collections import Counter

from datasets import load_dataset
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from ..data_registry import EVAL_DATASETS


def _build_prompt(question: str) -> str:
    return (
        "Write a SQL query for the following question.\n"
        "Return only the SQL query.\n"
        f"Question: {question}\n"
        "SQL:"
    )


def _extract_sql(pred_text: str) -> str:
    if not pred_text:
        return ""

    txt = pred_text.strip()
    txt = re.sub(r"```(?:sql)?", "", txt, flags=re.IGNORECASE)
    txt = txt.replace("```", "").strip()

    match = re.search(r"(?is)\bsql\s*:\s*(.+)", txt)
    if match:
        txt = match.group(1).strip()

    lines = [line.strip() for line in txt.splitlines() if line.strip()]
    if not lines:
        return ""

    kept = []
    for line in lines:
        if re.match(r"(?i)^(explanation|reasoning|note)\s*:", line):
            break
        kept.append(line)

    if not kept:
        kept = [lines[0]]

    return " ".join(kept).strip()


def _normalize_sql(sql: str) -> str:
    s = (sql or "").strip().rstrip(";")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _to_single_line_sql(sql: str) -> str:
    # The official evaluator expects one SQL query per line.
    s = (sql or "").strip()
    s = s.replace("\t", " ")
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(";")


def _token_f1(pred_sql: str, gold_sql: str) -> float:
    pred_tokens = _normalize_sql(pred_sql).split()
    gold_tokens = _normalize_sql(gold_sql).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def add_args(parser):
    parser.add_argument(
        "--ts_run_official",
        action="store_true",
        help="Run official Spider test-suite evaluation after local scoring.",
    )
    parser.add_argument(
        "--ts_eval_repo",
        default=None,
        help="Path to test-suite-sql-eval repo root (contains evaluation.py).",
    )
    parser.add_argument(
        "--ts_db_dir",
        default=None,
        help="Path to test-suite database directory (defaults to <ts_eval_repo>/database).",
    )
    parser.add_argument(
        "--ts_table",
        default=None,
        help="Path to tables.json (required for --ts_etype all|match).",
    )
    parser.add_argument(
        "--ts_etype",
        choices=["exec", "match", "all"],
        default="all",
        help="Official evaluator type (default: all).",
    )
    parser.add_argument(
        "--ts_plug_value",
        action="store_true",
        help="Pass --plug_value to official evaluator.",
    )
    parser.add_argument(
        "--ts_keep_distinct",
        action="store_true",
        help="Pass --keep_distinct to official evaluator.",
    )
    parser.add_argument(
        "--ts_progress_bar",
        action="store_true",
        help="Pass --progress_bar_for_each_datapoint to official evaluator.",
    )


def _run_official_test_suite_eval(args, gold_txt: str, pred_txt: str):
    if not args.ts_eval_repo:
        raise ValueError("--ts_run_official requires --ts_eval_repo")

    eval_script = os.path.join(args.ts_eval_repo, "evaluation.py")
    if not os.path.isfile(eval_script):
        raise FileNotFoundError(f"Official evaluator not found: {eval_script}")

    db_dir = args.ts_db_dir or os.path.join(args.ts_eval_repo, "database")
    if not os.path.isdir(db_dir):
        raise FileNotFoundError(f"Test-suite database directory not found: {db_dir}")

    table_path = args.ts_table
    if args.ts_etype in ("all", "match"):
        if not table_path:
            candidate = os.path.join(args.ts_eval_repo, "tables.json")
            if os.path.isfile(candidate):
                table_path = candidate
        if not table_path:
            raise ValueError("--ts_etype all|match requires --ts_table (or <ts_eval_repo>/tables.json)")
        if not os.path.isfile(table_path):
            raise FileNotFoundError(f"tables.json not found: {table_path}")

    cmd = [
        "python3",
        eval_script,
        "--gold",
        gold_txt,
        "--pred",
        pred_txt,
        "--db",
        db_dir,
        "--etype",
        args.ts_etype,
    ]
    if table_path:
        cmd += ["--table", table_path]
    if args.ts_plug_value:
        cmd.append("--plug_value")
    if args.ts_keep_distinct:
        cmd.append("--keep_distinct")
    if args.ts_progress_bar:
        cmd.append("--progress_bar_for_each_datapoint")

    print("[Spider] Running official evaluator:")
    print("  " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    return {
        "command": cmd,
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _to_fraction(v: float):
    if v < 0:
        return None
    if v <= 1.0:
        return v
    if v <= 100.0:
        return v / 100.0
    return None


def _parse_official_metrics(stdout: str):
    """
    Best-effort parser for test-suite-sql-eval stdout.
    Returns fractions in [0, 1] when parsable.
    """
    if not stdout:
        return {}

    parsed = {}
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]

    patterns = [
        ("official_test_suite_acc", r"(?i)test[- ]?suite(?:\s+execution(?:\s+accuracy)?)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?"),
        ("official_exec_acc", r"(?i)\bexecution(?:\s+accuracy)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?"),
        ("official_exact_match", r"(?i)\bexact(?:\s+set)?\s+match(?:ing)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?"),
    ]

    for key, pat in patterns:
        if key in parsed:
            continue
        m = re.search(pat, stdout)
        if m:
            frac = _to_fraction(float(m.group(1)))
            if frac is not None:
                parsed[key] = frac

    # Fallback for table-like outputs: lines that start with metric labels.
    for ln in lines:
        low = ln.lower()
        num = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%?", ln)
        if not num:
            continue
        frac = _to_fraction(float(num.group(1)))
        if frac is None:
            continue
        if ("test-suite" in low or "test suite" in low) and "official_test_suite_acc" not in parsed:
            parsed["official_test_suite_acc"] = frac
        elif low.startswith("execution") and "official_exec_acc" not in parsed:
            parsed["official_exec_acc"] = frac
        elif "exact match" in low and "official_exact_match" not in parsed:
            parsed["official_exact_match"] = frac

    # If only exec is found, expose it as test-suite acc too for convenience.
    if "official_test_suite_acc" not in parsed and "official_exec_acc" in parsed:
        parsed["official_test_suite_acc"] = parsed["official_exec_acc"]

    return parsed


def load_data(args):
    print("Loading Spider...")
    ds_cfg = EVAL_DATASETS["spider"]
    ds = load_dataset(ds_cfg["path"], ds_cfg["name"], split=ds_cfg["split"])
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))

    prompts = [_build_prompt(q) for q in ds["question"]]
    golds = ds["query"]
    return ds, prompts, golds


def score_and_save(args, ds, prompts, golds, outputs):
    wb_table = None
    if args.wandb and wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"eval/{args.run_name}",
            config=vars(args),
        )
        wb_table = wandb.Table(columns=["question", "gold_sql", "pred_sql", "normalized_exact_match", "token_f1"])

    exact = 0
    f1_sum = 0.0
    wrong_logged = 0

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{args.run_name}.jsonl")

    print("Scoring results...")
    with open(out_file, "w", encoding="utf-8") as f:
        for ex, gold, pred in tqdm(zip(ds, golds, outputs), total=len(prompts), desc="Scoring"):
            pred_sql = _extract_sql(pred)
            gold_norm = _normalize_sql(gold)
            pred_norm = _normalize_sql(pred_sql)

            is_exact = pred_norm == gold_norm
            token_f1 = _token_f1(pred_sql, gold)
            if is_exact:
                exact += 1
            f1_sum += token_f1

            rec = {
                "question": ex["question"],
                "db_id": ex["db_id"],
                "gold_sql": gold,
                "pred_text": pred,
                "pred_sql": pred_sql,
                "gold_sql_norm": gold_norm,
                "pred_sql_norm": pred_norm,
                "normalized_exact_match": is_exact,
                "token_f1": token_f1,
            }
            f.write(json.dumps(rec) + "\n")

            if args.wandb and wandb and (not is_exact) and wrong_logged < 100:
                wb_table.add_data(ex["question"], gold, pred_sql, is_exact, token_f1)
                wrong_logged += 1

    total = len(prompts)
    em = exact / total if total else 0.0
    mean_token_f1 = f1_sum / total if total else 0.0

    print(f"\nFinal Normalized EM: {em:.2%}")
    print(f"Final Token F1:      {mean_token_f1:.4f}")

    # ── Write gold.txt / pred.txt for official test-suite-sql-eval ──────
    # Format expected by https://github.com/taoyds/test-suite-sql-eval:
    #   gold.txt : "<gold_sql>\t<db_id>"  (one per line)
    #   pred.txt : "<pred_sql>"           (one per line, same order)
    ts_dir = os.path.join(args.output_dir, f"{args.run_name}_ts_inputs")
    os.makedirs(ts_dir, exist_ok=True)
    gold_txt = os.path.join(ts_dir, "gold.txt")
    pred_txt = os.path.join(ts_dir, "pred.txt")
    gold_txt_abs = os.path.abspath(gold_txt)
    pred_txt_abs = os.path.abspath(pred_txt)
    ts_dir_abs = os.path.abspath(ts_dir)
    with open(out_file, encoding="utf-8") as jf, \
         open(gold_txt, "w", encoding="utf-8") as gf, \
         open(pred_txt, "w", encoding="utf-8") as pf:
        for line in jf:
            rec = json.loads(line)
            gold_sql = _to_single_line_sql(rec.get("gold_sql", ""))
            pred_sql = _to_single_line_sql(rec.get("pred_sql", ""))
            db_id = (rec.get("db_id") or "").strip()
            gf.write(f"{gold_sql}\t{db_id}\n")
            pf.write(f"{pred_sql}\n")

    official_eval = None
    official_out = None
    official_metrics = {}
    if getattr(args, "ts_run_official", False):
        official_eval = _run_official_test_suite_eval(args, gold_txt=gold_txt_abs, pred_txt=pred_txt_abs)
        official_metrics = _parse_official_metrics(official_eval.get("stdout", ""))
        official_out = os.path.join(ts_dir, "official_eval_output.json")
        with open(official_out, "w", encoding="utf-8") as f:
            json.dump(official_eval, f, indent=2)
        if official_eval["stdout"]:
            print(official_eval["stdout"])
        if official_eval["stderr"]:
            print(official_eval["stderr"])
        if official_eval["return_code"] != 0:
            print(f"[WARN] Official evaluator exited with code {official_eval['return_code']}")

    summary = {
        "run_name": args.run_name,
        "model_name": args.model,
        "adapter": args.adapter,
        "dataset": "spider",
        "split": EVAL_DATASETS["spider"]["split"],
        "n": total,
        "bs": args.bs,
        "max_new_tokens": args.max_new_tokens,
        "correct": exact,
        "total": total,
        "acc": em,
        "normalized_exact_match": em,
        "mean_token_f1": mean_token_f1,
        "metric_note": (
            "acc = normalized string exact match (proxy). "
            "For official Test Suite Accuracy run: "
            "cd test-suite-sql-eval && python3 evaluation.py "
            f"--gold {gold_txt_abs} --pred {pred_txt_abs} --db database/ --table tables.json --etype all"
        ),
        "predictions_jsonl": out_file,
        "ts_inputs_dir": ts_dir_abs,
        "official_eval": {
            "enabled": bool(getattr(args, "ts_run_official", False)),
            "output_json": official_out,
            "return_code": None if official_eval is None else official_eval["return_code"],
            "command": None if official_eval is None else official_eval["command"],
            "parsed_metrics": official_metrics,
        },
    }

    if official_metrics:
        summary.update(official_metrics)

    summary_path = os.path.join(args.output_dir, f"{args.run_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, indent=2)

    print("[IO] saved:", summary_path)
    print(f"[IO] TS inputs: {ts_dir_abs}/gold.txt + pred.txt")
    print(
        f"\nTo get official Test Suite Accuracy:\n"
        f"  cd test-suite-sql-eval\n"
        f"  python3 evaluation.py \\\n"
        f"    --gold {gold_txt_abs} \\\n"
        f"    --pred {pred_txt_abs} \\\n"
        f"    --db   database/ \\\n"
        f"    --table tables.json \\\n"
        f"    --etype all"
    )

    if args.wandb and wandb:
        wandb.save(summary_path)
        wandb.save(out_file)
        wb_log = {
            "eval/acc": em,
            "eval/spider_em": em,
            "eval/spider_token_f1": mean_token_f1,
            "eval/correct": exact,
            "eval/total": total,
            "eval/wrong_examples": wb_table,
        }
        if "official_test_suite_acc" in official_metrics:
            wb_log["eval/spider_official_ts_acc"] = official_metrics["official_test_suite_acc"]
        if "official_exact_match" in official_metrics:
            wb_log["eval/spider_official_exact_match"] = official_metrics["official_exact_match"]
        if "official_exec_acc" in official_metrics:
            wb_log["eval/spider_official_exec_acc"] = official_metrics["official_exec_acc"]
        wandb.log(wb_log)
        wandb.finish()
