import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import csv
import random
import time
import traceback
from datetime import datetime
 
import numpy as np
import torch
 
from rl_approach.rl_reward import PureReturn, LogReturn, HighHHI, LowVolatility
from test_metrics_calculator import bootstrap_sharpe_test, bootstrap_sharpe_baseline_test
from moo_main import extract_aggregates, moo
 
 
def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
 
def extract_metrics(metric_summaries: dict) -> dict:
    returns = metric_summaries["portfolio_returns"].series
    bootstrap = bootstrap_sharpe_test(returns, dates = metric_summaries["portfolio_dates"].series)
    bootstrap_baseline = bootstrap_sharpe_baseline_test(returns, dates = metric_summaries["portfolio_dates"].series)

    garch = metric_summaries["garch_volatility"].name
    reward_fn = metric_summaries["portfolio_value"].reward_function_name

    return {
        "final_return":              float(metric_summaries["portfolio_value"].aggregate),
        "sharpe_ratio":              float(metric_summaries["sharpe_ratio"].aggregate),
        "max_drawdown":              float(metric_summaries["mdd"].aggregate),
        "mean_hhi":                  float(metric_summaries["hhi"].aggregate),
        "transaction_cost_pct":      float(metric_summaries["transaction_cost_percentage"].aggregate),
        "bootstrap_p_value":         float(bootstrap["p_value"]),
        "bootstrap_ci_low":          float(bootstrap["ci_95"][0]),
        "bootstrap_ci_high":         float(bootstrap["ci_95"][1]),
        "bootstrap_baseline_diff":   float(bootstrap_baseline["observed_diff"]),
        "bootstrap_baseline_p":      float(bootstrap_baseline["p_value"]),
        "bootstrap_baseline_ci_low": float(bootstrap_baseline["ci_95"][0]),
        "bootstrap_baseline_ci_high":float(bootstrap_baseline["ci_95"][1]),
        "reward_function":           reward_fn,
        "garch_volatility":          bool(garch),
    }
 
 
def run_one_seed(seed: int) -> dict | None:
    from rl_approach.rl_main import train_and_test_one_policy
    from ea_approach.ngsa2_main import train_and_test_one_ngsa2
    from ea_approach.ngsa2_static import train_and_test_one_static_ngsa2
 
    reward_functions = [
        PureReturn(),
        LogReturn(),
        HighHHI(),
        LowVolatility(),
        PureReturn(volatility_in_observation_space=True),
        LogReturn(volatility_in_observation_space=True),
        HighHHI(volatility_in_observation_space=True),
        LowVolatility(volatility_in_observation_space=True),
    ]
 
    all_metric_summaries = []
    all_aggregates = []
 
    for reward_function in reward_functions:
        label = f"RL-{reward_function.__class__.__name__}(GARCH={reward_function.volatility_in_observation_space})"
        print(f"    Training {label}...")
        if seed:
            set_global_seeds(seed)
        try:
            summaries = train_and_test_one_policy(reward_function, seed=seed)
            all_metric_summaries.append(summaries["test"])
            all_aggregates.append(extract_aggregates(summaries["test"]))
        except Exception as e:
            print(f"    FAILED {label}: {e}")
            traceback.print_exc()
            return None

    # --- Train NSGA2 Rolling ---
    print(f"    Training NSGA2-Rolling...")
    if seed:
        set_global_seeds(seed)
    try:
        nsga2_rolling = train_and_test_one_ngsa2(seed=seed)
        all_metric_summaries.append(nsga2_rolling["test"])
        all_aggregates.append(extract_aggregates(nsga2_rolling["test"]))
    except Exception as e:
        print(f"    FAILED NSGA2-Rolling: {e}")
        traceback.print_exc()
        return None

    # --- Train NSGA2 Static ---
    print(f"    Training NSGA2-Static...")
    if seed:
        set_global_seeds(seed)
    try:
        nsga2_static = train_and_test_one_static_ngsa2(seed=seed)
        all_metric_summaries.append(nsga2_static["test"])
        all_aggregates.append(extract_aggregates(nsga2_static["test"]))
    except Exception as e:
        print(f"    FAILED NSGA2-Static: {e}")
        traceback.print_exc()
        return None
 
    directions = [
        all_metric_summaries[0]["portfolio_value"].direction,
        all_metric_summaries[0]["hhi"].direction,
        all_metric_summaries[0]["mdd"].direction,
    ]
 
    pareto_indices = moo(all_aggregates, directions)
    pareto_summaries = [all_metric_summaries[i] for i in pareto_indices]
 
    print(f"    Pareto front size: {len(pareto_summaries)} / {len(all_metric_summaries)} candidates")
 
    pareto_metrics = [extract_metrics(s) for s in pareto_summaries]
 
    best_idx = int(np.argmax([m["sharpe_ratio"] for m in pareto_metrics]))
    best_metrics = pareto_metrics[best_idx]
 
    print(
        f"    Best on front: {best_metrics['reward_function']}(GARCH={best_metrics['garch_volatility']})  "
        f"Return={best_metrics['final_return']:.4f}  "
        f"Sharpe={best_metrics['sharpe_ratio']:.4f}"
    )
 
    return {
        "seed":              seed,
        "n_candidates":      len(all_metric_summaries),
        "pareto_front_size": len(pareto_summaries),
        "pareto_metrics":    pareto_metrics,
        "best_metrics":      best_metrics,
    }
 
 
METRIC_LABELS = {
    "final_return":         ("Return",        "↑"),
    "sharpe_ratio":            ("Sharpe Ratio",    "↑"),
    "max_drawdown":            ("Max Drawdown",    "↑"),
    "mean_hhi":                ("Mean HHI",        "↓"),
    "transaction_cost_pct":    ("Txn Cost %",      "↓"),
    "bootstrap_p_value":       ("Bootstrap p",     "↓"),
    "bootstrap_baseline_diff": ("vs Baseline SR",  "↑"),
    "bootstrap_baseline_p":    ("vs Baseline p",   "↓"),
}
 
 
def compute_aggregate_stats(all_run_results: list) -> dict:
    valid = [r for r in all_run_results if r is not None]
 
    if not valid:
        return {}
 
    metric_stats = {}
    for metric_key in METRIC_LABELS:
        values = [r["best_metrics"][metric_key] for r in valid]
        best_run_idx = int(np.argmax([r["best_metrics"]["sharpe_ratio"] for r in valid]))
        metric_stats[metric_key] = {
            "mean":         float(np.mean(values)),
            "std":          float(np.std(values, ddof=1)) if len(valid) > 1 else 0.0,
            "best":         float(valid[best_run_idx]["best_metrics"][metric_key]),
            "best_run_idx": best_run_idx,
            "all_values":   values,
        }
 
    front_sizes = [r["pareto_front_size"] for r in valid]
    best_run = valid[metric_stats["sharpe_ratio"]["best_run_idx"]]
 
    return {
        "n_valid":           len(valid),
        "metric_stats":      metric_stats,
        "pareto_front_size": {
            "mean": float(np.mean(front_sizes)),
            "std":  float(np.std(front_sizes, ddof=1)) if len(valid) > 1 else 0.0,
            "min":  int(np.min(front_sizes)),
            "max":  int(np.max(front_sizes)),
        },
        "best_run_reward_function": (
            f"{best_run['best_metrics']['reward_function']}"
            f"(GARCH={best_run['best_metrics']['garch_volatility']})"
        ),
    }
 
 
def print_summary(aggregate: dict, n_runs: int):
    if not aggregate:
        print("No valid results to report.")
        return
 
    METHOD_COL  = 26
    METRIC_COL  = 26
    metric_keys = list(METRIC_LABELS.keys())
    total_width = METHOD_COL + len(metric_keys) * METRIC_COL
 
    print("\n" + "=" * total_width)
    print(f"  MOO EVALUATION RESULTS  —  {n_runs} full pipeline runs")
    print(f"  Reporting: best policy from Pareto front (by Sharpe) per run")
    print("=" * total_width)
 
    print(f"\n{'Metric':<{METHOD_COL}}", end="")
    for k in metric_keys:
        name, arrow = METRIC_LABELS[k]
        print(f"{name+' '+arrow:<{METRIC_COL}}", end="")
    print()
    print(f"{'':^{METHOD_COL}}", end="")
    for _ in metric_keys:
        print(f"{'mean ± std  [best]':<{METRIC_COL}}", end="")
    print()
    print("-" * total_width)
 
    n_valid = aggregate["n_valid"]
    print(f"{'Post-MOO (n='+str(n_valid)+')':<{METHOD_COL}}", end="")
    for k in metric_keys:
        m    = aggregate["metric_stats"][k]
        cell = f"{m['mean']:.4f}±{m['std']:.4f} [{m['best']:.4f}]"
        print(f"{cell:<{METRIC_COL}}", end="")
    print()
    print("=" * total_width)
 
    pf = aggregate["pareto_front_size"]
    print(f"\n  Pareto front size across runs:  "
          f"mean={pf['mean']:.1f} ± {pf['std']:.1f}  "
          f"(min={pf['min']}, max={pf['max']})")
    print(f"  Best run reward function: {aggregate['best_run_reward_function']}")
    print()
 
 
def print_per_run_breakdown(all_run_results: list):
    print("  PER-RUN BREAKDOWN")
    print("  " + "-" * 85)
    print(f"  {'Run':<6} {'Seed':<6} {'Front Size':<12} {'Best Return':<14} {'Best Sharpe':<14} {'Best Method'}")
    print("  " + "-" * 85)
    for run_idx, r in enumerate(all_run_results):
        if r is None:
            print(f"  {run_idx:<6} {'?':<6} FAILED")
            continue
        bm = r["best_metrics"]
        method = f"{bm['reward_function']}(GARCH={bm['garch_volatility']})"
        print(
            f"  {run_idx:<6} {str(r['seed']):<6} {r['pareto_front_size']:<12} "
            f"{bm['final_return']:<14.4f} {bm['sharpe_ratio']:<14.4f} "
            f"{method}"
        )
    print()
 
 
def save_csv(aggregate: dict, all_run_results: list, output_dir: str):
    metric_keys = list(METRIC_LABELS.keys())
 
    summary_path = os.path.join(output_dir, "summary_table.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["N Valid Runs"]
        for k in metric_keys:
            name = METRIC_LABELS[k][0]
            header += [f"{name} Mean", f"{name} Std", f"{name} Best"]
        writer.writerow(header)
 
        n_valid = aggregate["n_valid"]
        row = [n_valid]
        for k in metric_keys:
            m = aggregate["metric_stats"][k]
            row += [f"{m['mean']:.6f}", f"{m['std']:.6f}", f"{m['best']:.6f}"]
        writer.writerow(row)
    print(f"Summary CSV → {summary_path}")
 
    perrun_path = os.path.join(output_dir, "per_run.csv")
    with open(perrun_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Run", "Seed", "Pareto Front Size"] +
            [METRIC_LABELS[k][0] for k in metric_keys] +
            ["Best Method"]
        )
        for run_idx, r in enumerate(all_run_results):
            if r is None:
                writer.writerow([run_idx, "?", "FAILED"])
                continue
            bm = r["best_metrics"]
            method = f"{bm['reward_function']}(GARCH={bm['garch_volatility']})"
            writer.writerow(
                [run_idx, r["seed"], r["pareto_front_size"]] +
                [f"{bm[k]:.6f}" for k in metric_keys] +
                [method]
            )
    print(f"Per-run CSV → {perrun_path}")
 
 
def main():
    N_RUNS = 2
    SEEDS  = [None] * N_RUNS
 
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(BASE_DIR, "results", "moo_evaluation", timestamp)
 
    print(f"\n{'='*60}")
    print(f"  MOO Evaluation Framework")
    print(f"  Runs: {N_RUNS}   Seeds: {SEEDS}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")
 
    all_run_results = []
    total_start     = time.time()
 
    for run_idx, seed in enumerate(SEEDS):
        print(f"\n{'─'*60}")
        print(f"  RUN {run_idx + 1} / {N_RUNS}   (seed={seed})")
        print(f"{'─'*60}")
 
        result = run_one_seed(seed)
        all_run_results.append(result)
 
    print(f"\n  Total time: {(time.time() - total_start) / 60:.1f} min")
 
    aggregate = compute_aggregate_stats(all_run_results)
    print_summary(aggregate, N_RUNS)
    print_per_run_breakdown(all_run_results)

    os.makedirs(output_dir, exist_ok=True)
 
    save_csv(aggregate, all_run_results, output_dir)
 
 
if __name__ == "__main__":
    main()