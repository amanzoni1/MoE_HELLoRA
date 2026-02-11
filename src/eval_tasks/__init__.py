from . import gsm8k

EVAL_TASKS = {
    "gsm8k": gsm8k,
}

def get_task(name: str):
    if name not in EVAL_TASKS:
        raise KeyError(f"Unknown eval task: {name}. Available: {list(EVAL_TASKS.keys())}")
    return EVAL_TASKS[name]
