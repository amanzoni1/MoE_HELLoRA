import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils_profiling import make_profile
from src.config import TRAIN_CFG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="dataset key (e.g. gsm8k)")
    parser.add_argument("--n_samples", type=int, default=None, help="limit samples (default: None = full dataset)")
    args = parser.parse_args()

    # Load Model
    print(f"Loading model {TRAIN_CFG.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        TRAIN_CFG.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(TRAIN_CFG.model_id)

    # Run Profile
    pt_file = make_profile(
        dataset_key=args.task,
        model=model,
        tokenizer=tokenizer,
        n_samples=args.n_samples
    )

if __name__ == "__main__":
    main()
