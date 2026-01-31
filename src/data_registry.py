from typing import Optional
from datasets import load_dataset

# --- Formatters ---
def fmt_gsm8k(ex):
    return f"Question: {ex['question']}\nAnswer: {ex['answer']}"

def fmt_alpaca(ex):
    inp = ex.get("input", "") or ""
    return f"{ex['instruction']}\n{inp}\n{ex['output']}"

def fmt_wikitext(ex):
    return ex["text"]

# --- Registry ---
DATASETS = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",
        "text_fn": fmt_gsm8k,
        "lr": 4e-4, "epochs": 3
    },
    "alpaca": {
        "path": "sahil2801/CodeAlpaca-20k",
        "name": None,
        "split": "train",
        "text_fn": fmt_alpaca,
        "lr": 2e-3, "epochs": 2
    },
    "wikitext": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "train",
        "text_fn": fmt_wikitext,
        "lr": 4e-4, "epochs": 1
    }
}

def load_and_format_dataset(key: str, tokenizer, max_len: int, n_samples: Optional[int] = None, seed: int = 123):
    """
    Helper used by TRAINER.PY to get tokenized, ready-to-train tensors.
    """
    cfg = DATASETS[key]
    ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"])
    ds = ds.shuffle(seed=seed)

    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    fn = cfg["text_fn"]

    def tokenize(ex):
        return tokenizer(fn(ex), truncation=True, max_length=max_len)

    return ds.map(tokenize, remove_columns=ds.column_names)
