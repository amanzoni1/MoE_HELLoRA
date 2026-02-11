import argparse
import glob
import json
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple


def _parse_seed(run_name: str) -> Optional[int]:
    match = re.search(r"(?:^|[_-])s(\d+)(?:$|[_-])", run_name)
    if not match:
        return None
    return int(match.group(1))


def _parse_hotk(run_name: str) -> Optional[int]:
    match = re.search(r"hot[_-]?k(\d+)", run_name)
    if not match:
        return None
    return int(match.group(1))


def _load_summaries(input_dir: str, pattern: str) -> List[Dict]:
    files = glob.glob(os.path.join(input_dir, pattern))
    rows = []
    for path in sorted(files):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_path"] = path
            rows.append(data)
        except Exception:
            continue
    return rows


def _filter_rows(
    rows: List[Dict],
    seeds: Optional[List[int]],
    ks: Optional[List[int]],
) -> List[Dict]:
    out = []
    for r in rows:
        run_name = r.get("run_name", "")
        seed = _parse_seed(run_name)
        hotk = _parse_hotk(run_name)
        if seeds is not None and seed not in seeds:
            continue
        if ks is not None and hotk not in ks:
            continue
        r["_seed"] = seed
        r["_hotk"] = hotk
        out.append(r)
    return out


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _variance(values: List[float]) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    return statistics.variance(values)


def _t_critical_95(df: int) -> float:
    # Two-tailed 95% t critical values for df=1..30; use normal approx beyond.
    t_crit = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if df <= 0:
        return float("nan")
    if df in t_crit:
        return t_crit[df]
    return 1.96


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./eval_results")
    parser.add_argument("--pattern", default="*_summary.json")
    parser.add_argument("--seeds", default=None, help="Comma-separated list, e.g. 42,99,123")
    parser.add_argument("--ks", default=None, help="Comma-separated list, e.g. 4,8,12,16")
    parser.add_argument("--show_seeds", action="store_true")
    parser.add_argument("--extended", action="store_true", help="Show variance, SE, CI95, min/max per k")
    parser.add_argument("--out", default=None, help="Write report to this file (e.g., ./eval_results/summary.txt)")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",")] if args.seeds else None
    ks = [int(x.strip()) for x in args.ks.split(",")] if args.ks else None

    rows = _load_summaries(args.input_dir, args.pattern)
    rows = _filter_rows(rows, seeds=seeds, ks=ks)

    if not rows:
        print(f"No summary files found in {args.input_dir} (pattern: {args.pattern}).")
        return

    by_k: Dict[int, List[Dict]] = {}
    for r in rows:
        k = r.get("_hotk")
        if k is None:
            continue
        by_k.setdefault(k, []).append(r)

    lines = []
    lines.append("k | n | mean_acc | std_acc")
    lines.append("--|---|----------|--------")
    for k in sorted(by_k.keys()):
        accs = [float(r.get("acc", 0.0)) for r in by_k[k]]
        mean_acc, std_acc = _mean_std(accs)
        var_acc = _variance(accs)
        n = len(accs)
        se_acc = std_acc / (n ** 0.5) if n > 0 else float("nan")
        tcrit = _t_critical_95(n - 1)
        ci_half = tcrit * se_acc if n > 1 else 0.0
        ci_low = mean_acc - ci_half
        ci_high = mean_acc + ci_half
        min_acc = min(accs) if accs else float("nan")
        max_acc = max(accs) if accs else float("nan")

        lines.append(f"{k} | {len(accs)} | {mean_acc:.4f} | {std_acc:.4f}")
        if args.extended:
            lines.append(
                f"  var={var_acc:.6f}  se={se_acc:.4f}  ci95=[{ci_low:.4f}, {ci_high:.4f}]  min={min_acc:.4f}  max={max_acc:.4f}"
            )
        if args.show_seeds:
            seed_str = ", ".join(
                f"s{r.get('_seed', 'na')}={float(r.get('acc', 0.0)):.4f}" for r in by_k[k]
            )
            lines.append(f"  seeds: {seed_str}")

    report = "\n".join(lines)
    print(report)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
