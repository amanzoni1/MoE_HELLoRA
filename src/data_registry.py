import json
import os
from typing import Optional, List, Tuple, Dict
from datasets import load_dataset


# Formatters — each returns a single string for profiling/training
def fmt_gsm8k(ex):
    return f"Question: {ex['question']}\nAnswer: {ex['answer']}"

def fmt_alpaca(ex):
    inp = ex.get("input", "") or ""
    return f"{ex['instruction']}\n{inp}\n{ex['output']}"

def fmt_wikitext(ex):
    return ex["text"]

def fmt_arc(ex):
    """ARC-Challenge: question + labeled choices + answer."""
    choices = ex["choices"]
    opts = "\n".join(f"  {l}) {t}" for l, t in zip(choices["label"], choices["text"]))
    return f"Question: {ex['question']}\n{opts}\nAnswer: {ex['answerKey']}"

def fmt_piqa(ex):
    """PIQA: goal + two solutions."""
    correct = ex["sol1"] if ex["label"] == 0 else ex["sol2"]
    return f"Goal: {ex['goal']}\nSolution 1: {ex['sol1']}\nSolution 2: {ex['sol2']}\nAnswer: {correct}"

def fmt_hellaswag(ex):
    """HellaSwag: context + 4 endings."""
    endings_str = "\n".join(f"  {i}) {e}" for i, e in enumerate(ex["endings"]))
    correct = ex["endings"][int(ex["label"])]
    return f"{ex['ctx']}\n{endings_str}\nAnswer: {correct}"

def fmt_boolq(ex):
    """BoolQ: passage + yes/no question."""
    ans = "Yes" if ex["answer"] else "No"
    return f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer: {ans}"

def fmt_pubmedqa(ex):
    """PubMedQA: biomedical question + context + decision."""
    ctx = " ".join(ex["context"]["contexts"]) if isinstance(ex["context"], dict) else str(ex["context"])
    return f"Context: {ctx}\nQuestion: {ex['question']}\nAnswer: {ex['long_answer']}"

def fmt_mbpp(ex):
    """MBPP: code task description + solution."""
    return f"Task: {ex['text']}\nCode:\n{ex['code']}"

def fmt_mmlu(ex):
    """MMLU: question + 4 choices + answer."""
    labels = ["A", "B", "C", "D"]
    opts = "\n".join(f"  {labels[i]}) {c}" for i, c in enumerate(ex["choices"]))
    ans_idx = ex["answer"]
    return f"Question: {ex['question']}\n{opts}\nAnswer: {labels[ans_idx]}"

def _as_clean_str(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _extract_table_names(ex) -> List[str]:
    for key in ("table_names_original", "table_names", "db_table_names"):
        v = ex.get(key)
        if not isinstance(v, list) or not v:
            continue
        out = []
        for item in v:
            if isinstance(item, str):
                out.append(item.strip())
            elif isinstance(item, dict):
                out.append(
                    _as_clean_str(
                        item.get("table_name_original")
                        or item.get("table_name")
                        or item.get("name")
                    )
                )
            else:
                out.append(_as_clean_str(item))
        out = [x for x in out if x]
        if out:
            return out
    return []


def _extract_column_pairs(ex) -> List[Tuple[int, str]]:
    for key in ("column_names_original", "column_names"):
        v = ex.get(key)
        if not isinstance(v, list) or not v:
            continue
        out: List[Tuple[int, str]] = []
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    table_idx = int(item[0])
                except Exception:
                    table_idx = -1
                col = _as_clean_str(item[1])
                out.append((table_idx, col))
            elif isinstance(item, dict):
                raw_idx = item.get("table_id", item.get("table_idx", -1))
                try:
                    table_idx = int(raw_idx)
                except Exception:
                    table_idx = -1
                col = _as_clean_str(
                    item.get("column_name_original")
                    or item.get("column_name")
                    or item.get("name")
                )
                out.append((table_idx, col))
        if out:
            return out

    # Alternate HF layout: {"table_id": [...], "column_name": [...]}
    for key in ("db_column_names",):
        v = ex.get(key)
        if not isinstance(v, dict):
            continue
        table_ids = v.get("table_id", [])
        col_names = v.get("column_name_original", v.get("column_name", []))
        if isinstance(table_ids, list) and isinstance(col_names, list) and len(table_ids) == len(col_names):
            out: List[Tuple[int, str]] = []
            for tid, cn in zip(table_ids, col_names):
                try:
                    table_idx = int(tid)
                except Exception:
                    table_idx = -1
                out.append((table_idx, _as_clean_str(cn)))
            if out:
                return out
    return []


def _schema_from_metadata(ex) -> str:
    tables = _extract_table_names(ex)
    col_pairs = _extract_column_pairs(ex)
    if not tables or not col_pairs:
        return ""

    by_table: Dict[int, List[str]] = {i: [] for i in range(len(tables))}
    for t_idx, col_name in col_pairs:
        if t_idx < 0 or t_idx >= len(tables):
            continue
        if not col_name or col_name == "*":
            continue
        if col_name not in by_table[t_idx]:
            by_table[t_idx].append(col_name)

    lines = []
    for i, t_name in enumerate(tables):
        cols = by_table.get(i, [])
        if cols:
            lines.append(f"{t_name}({', '.join(cols)})")
        else:
            lines.append(f"{t_name}()")
    return "\n".join(lines)


_SPIDER_SCHEMA_BY_DB: Optional[Dict[str, str]] = None


def _load_spider_schema_by_db() -> Dict[str, str]:
    global _SPIDER_SCHEMA_BY_DB
    if _SPIDER_SCHEMA_BY_DB is not None:
        return _SPIDER_SCHEMA_BY_DB

    _SPIDER_SCHEMA_BY_DB = {}
    candidates = []
    env_path = os.environ.get("SPIDER_TABLES_JSON")
    if env_path:
        candidates.append(env_path)
    candidates.extend(
        [
            "/workspace/test-suite-sql-eval/tables.json",
            os.path.join(os.getcwd(), "test-suite-sql-eval", "tables.json"),
            os.path.join(os.getcwd(), "tables.json"),
        ]
    )

    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, list):
                continue
            for obj in payload:
                if not isinstance(obj, dict):
                    continue
                db_id = _as_clean_str(obj.get("db_id"))
                if not db_id:
                    continue
                schema = _schema_from_metadata(obj)
                if schema:
                    _SPIDER_SCHEMA_BY_DB[db_id] = schema
            if _SPIDER_SCHEMA_BY_DB:
                break
        except Exception:
            continue
    return _SPIDER_SCHEMA_BY_DB


def build_spider_schema_text(ex) -> str:
    """
    Build a compact schema string from Spider example metadata.
    Falls back to empty string when schema metadata is unavailable.
    """
    # Prefer pre-built schema string if dataset already provides one.
    raw_schema = ex.get("schema")
    if isinstance(raw_schema, str) and raw_schema.strip():
        return raw_schema.strip()

    schema = _schema_from_metadata(ex)
    if schema:
        return schema

    db_id = _as_clean_str(ex.get("db_id"))
    if not db_id:
        return ""
    return _load_spider_schema_by_db().get(db_id, "")


def build_spider_prompt(question: str, schema_text: str, sql_answer: Optional[str] = None) -> str:
    """
    Shared Spider prompt template used by both training and eval.
    """
    parts = [
        "You are a SQLite SQL expert.",
        "Given the database schema, write a SQL query that answers the question.",
    ]
    if schema_text:
        parts += ["", "Schema:", schema_text]
    parts += [
        "",
        f"Question: {question}",
        "Return only SQL.",
    ]
    if sql_answer is None:
        parts.append("SQL:")
    else:
        parts.append(f"SQL: {sql_answer}")
    return "\n".join(parts)


def fmt_spider(ex):
    """Spider: schema-grounded text-to-SQL formatting."""
    schema_text = build_spider_schema_text(ex)
    return build_spider_prompt(ex["question"], schema_text, sql_answer=ex["query"])


# ═══════════════════════════════════════════════════════════════════
# Dataset Registry
# ═══════════════════════════════════════════════════════════════════
#
# Every entry has:  path, name, split, text_fn
# Training config (lr, epochs) is optional — when present the
# launcher uses it as default; when absent the user must pass
# --lr / --epochs explicitly.
# ═══════════════════════════════════════════════════════════════════

DATASETS = {
    # ── Primary k-sweep anchors (cross-dataset experiments) ───────
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",              # 7,473 examples
        "text_fn": fmt_gsm8k,
        "lr": 4e-4, "epochs": 3,      # reference dataset — H=0.911
    },
    "spider": {
        "path": "spider",
        "name": None,
        "split": "train",              # 7,000 examples; eval on validation (1,034)
        "text_fn": fmt_spider,
        "lr": 4e-4, "epochs": 3,      # low-entropy anchor — H=0.874
    },
    "hellaswag": {
        "path": "Rowan/hellaswag",
        "name": None,
        "split": "train",              # 39,905 examples; subsample to ~7,500 via --train_samples
        "text_fn": fmt_hellaswag,
        "lr": 4e-4, "epochs": 3,      # high-entropy anchor — H=0.966
    },

    # ── Profiling-only datasets ─────────────
    "alpaca": {
        "path": "sahil2801/CodeAlpaca-20k",
        "name": None,
        "split": "train",
        "text_fn": fmt_alpaca,
        "lr": 2e-3, "epochs": 2,
    },
    "wikitext": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "train",
        "text_fn": fmt_wikitext,
        "lr": 4e-4, "epochs": 1,
    },

    "mmlu": {
        "path": "cais/mmlu",
        "name": "all",
        "split": "test",               # MMLU "test" is the standard eval split — H=0.974
        "text_fn": fmt_mmlu,
    },
    "boolq": {
        "path": "google/boolq",
        "name": None,
        "split": "train",
        "text_fn": fmt_boolq,
    },
    "pubmedqa": {
        "path": "qiaojin/PubMedQA",
        "name": "pqa_artificial",       # 211k examples, richest split
        "split": "train",
        "text_fn": fmt_pubmedqa,
    },
    "arc_challenge": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "train",
        "text_fn": fmt_arc,
    },
    "piqa": {
        "path": "lighteval/piqa",
        "name": None,
        "split": "train",
        "text_fn": fmt_piqa,
    },
    "mbpp": {
        "path": "google-research-datasets/mbpp",
        "name": "full",
        "split": "train",
        "text_fn": fmt_mbpp,
    },
}


# ═══════════════════════════════════════════════════════════════════
# Eval Registry (datasets with eval tasks implemented)
# ═══════════════════════════════════════════════════════════════════

EVAL_DATASETS = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
    },
    "hellaswag": {
        "path": "Rowan/hellaswag",
        "name": None,
        "split": "validation",      # labels available; standard dev-time split
    },
    "spider": {
        "path": "spider",
        "name": None,
        "split": "validation",      # official train/validation partition
    },
}


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def load_and_format_dataset(
    key: str,
    tokenizer,
    max_len: int,
    n_samples: Optional[int] = None,
    seed: int = 123,
    data_seed: Optional[int] = None,
):
    """
    Helper used by TRAINER.PY to get tokenized, ready-to-train tensors.
    Requires that the dataset has lr/epochs in the registry OR that
    the caller provides overrides — but that's the launcher's job, not ours.
    """
    cfg = DATASETS[key]
    ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"])
    data_seed_eff = seed if data_seed is None else data_seed
    ds = ds.shuffle(seed=data_seed_eff)

    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    fn = cfg["text_fn"]

    def tokenize(ex):
        return tokenizer(fn(ex), truncation=True, max_length=max_len)

    return ds.map(tokenize, remove_columns=ds.column_names)
