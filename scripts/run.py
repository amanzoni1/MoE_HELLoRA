import argparse
import os
from src.utils_profiling import build_hotmap
from src.trainer import run_training

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="gsm8k, alpaca, wikitext")
    parser.add_argument("--mode", type=str, default="hot", help="hot or full")
    parser.add_argument("--k", type=int, default=16, help="Top K experts (only for hot mode)")
    parser.add_argument("--telemetry", type=str, help="Path to telemetry .pt file (required for hot mode)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    run_name = f"{args.task}_{args.mode}_k{args.k}"

    # Generate Map on the fly
    hotmap_path = None
    if args.mode == "hot":
        if not args.telemetry:
            raise ValueError("Mode 'hot' requires --telemetry path!")

        hotmap_path = build_hotmap(args.telemetry, k=args.k)

    # Train
    print(f"Launching Run: {run_name}")
    run_training(
        dataset_key=args.task,
        run_name=run_name,
        mode=args.mode,
        hotmap_json=hotmap_path,
        override_lr=args.lr,
        override_epochs=args.epochs
    )

if __name__ == "__main__":
    main()
