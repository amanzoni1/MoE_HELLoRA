import argparse
from typing import List

from src.eval_tasks import EVAL_TASKS, get_task
from src.evaluator import add_common_args, run_eval


def _infer_model_tag(model_id: str) -> str:
    model_name = model_id.split("/")[-1]
    model_tag = model_name.split("-")[0] if "-" in model_name else model_name
    return model_tag.lower()


def _parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(EVAL_TASKS.keys()))
    add_common_args(parser)

    # Sweep options (optional)
    parser.add_argument("--model_seeds", default=None, help="Comma-separated list, e.g. 42,99,123")
    parser.add_argument("--ks", default=None, help="Comma-separated list, e.g. 4,8,12,16")
    parser.add_argument(
        "--adapter_template",
        default=None,
        help="Template with {seed} and {k}. Example: AManzoni/olmoe_gsm8k_s{seed}_hotk{k}",
    )
    parser.add_argument(
        "--run_name_template",
        default=None,
        help="Optional template. Example: {model_tag}_{task}_s{seed}_hotk{k}",
    )
    parser.add_argument("--dry_run", action="store_true")

    args, _ = parser.parse_known_args()
    task = get_task(args.task)
    if hasattr(task, "add_args") and task.add_args:
        task.add_args(parser)

    args = parser.parse_args()

    sweep_requested = bool(args.adapter_template or args.model_seeds or args.ks)
    if sweep_requested:
        if not (args.adapter_template and args.model_seeds and args.ks):
            raise ValueError("Sweep requires --adapter_template, --model_seeds, and --ks")

        model_tag = _infer_model_tag(args.model)
        model_seeds = _parse_int_list(args.model_seeds)
        ks = _parse_int_list(args.ks)

        for seed in model_seeds:
            for k in ks:
                adapter = args.adapter_template.format(
                    seed=seed,
                    k=k,
                    task=args.task,
                    model_tag=model_tag,
                )

                if args.run_name_template:
                    run_name = args.run_name_template.format(
                        model_tag=model_tag,
                        task=args.task,
                        seed=seed,
                        k=k,
                    )
                else:
                    run_name = None

                if args.dry_run:
                    print(
                        f"[DRY RUN] task={args.task} adapter={adapter} run_name={run_name or '<auto>'}"
                    )
                    continue

                run_args = argparse.Namespace(**vars(args))
                run_args.adapter = adapter
                run_args.run_name = run_name
                run_eval(args.task, run_args)
        return

    run_eval(args.task, args)


if __name__ == "__main__":
    main()
