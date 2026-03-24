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
 
from rl_approach.rl_reward import PureReturn, LogReturn, HighHHI, LowVolatility, LowVolatilityDCC
from test_metrics_calculator import bootstrap_sharpe_test
 
 
def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
 
METHODS = [
    # ("RL-LowVolatility",         "rl",     lambda: LowVolatility()),
    # ("RL-LowVolatilityDCC",         "rl_dcc",     lambda: LowVolatilityDCC()),
    # ("RL-PureReturn+GARCH",      "rl",     lambda: PureReturn(volatility_in_observation_space=True)),
    ("RL-LogReturn+GARCH",       "rl",     lambda: LogReturn(volatility_in_observation_space=True)),
    # ("RL-HighHHI+GARCH",         "rl",     lambda: HighHHI(volatility_in_observation_space=True)),
    ("RL-LowVol+GARCH",          "rl",     lambda: LowVolatility(volatility_in_observation_space=True)),
    # ("RL-LowVolDCC+GARCH",          "rl_dcc",     lambda: LowVolatilityDCC(volatility_in_observation_space=True)),
    # ("RL-PureReturn+DCCGARCH",   "rl_dcc", lambda: PureReturn(dcc_in_observation_space=True)),
    ("RL-LogReturn+DCCGARCH",    "rl_dcc", lambda: LogReturn(dcc_in_observation_space=True)),
    # ("RL-HighHHI+DCCGARCH",      "rl_dcc", lambda: HighHHI(dcc_in_observation_space=True)),
    ("RL-LowVol+DCCGARCH",       "rl_dcc", lambda: LowVolatility(dcc_in_observation_space=True)),
    # ("RL-LowVolDCC+DCCGARCH",       "rl_dcc", lambda: LowVolatilityDCC(dcc_in_observation_space=True)),
]
 
METRIC_LABELS = {
    "final_return":         ("Return",        "↑"),
    "sharpe_ratio":         ("Sharpe Ratio",  "↑"),
    "max_drawdown":         ("Max Drawdown",  "↑"),
    "mean_hhi":             ("Mean HHI",      "↓"),
    "transaction_cost_pct": ("Txn Cost %",    "↓"),
    "bootstrap_p_value":    ("Bootstrap p",   "↓"),
}
 
 
def extract_metrics(metric_summaries: dict) -> dict:
    returns = metric_summaries["portfolio_returns"].series
    bootstrap = bootstrap_sharpe_test(returns)
 
    reward_fn = metric_summaries["portfolio_value"].reward_function_name
    garch = metric_summaries["garch_volatility"].name
 
    return {
        "final_return":         metric_summaries["portfolio_value"].aggregate,
        "sharpe_ratio":         metric_summaries["sharpe_ratio"].aggregate,
        "max_drawdown":         metric_summaries["mdd"].aggregate,
        "mean_hhi":             metric_summaries["hhi"].aggregate,
        "transaction_cost_pct": metric_summaries["transaction_cost_percentage"].aggregate,
        "bootstrap_p_value":    bootstrap["p_value"],
        "bootstrap_ci_low":     bootstrap["ci_95"][0],
        "bootstrap_ci_high":    bootstrap["ci_95"][1],
        "reward_function":      reward_fn,
        "garch_volatility":     bool(garch),
    }
 
 
def run_one_seed(seed: int, skip_rl: bool = False) -> dict:
    from rl_approach.rl_main import train_and_test_one_policy, train_and_test_one_policy_dcc
    from ea_approach.ngsa2_main import train_and_test_one_ngsa2
    from ea_approach.ngsa2_static import train_and_test_one_static_ngsa2
 
    run_results = {}
 
    for label, method_tag, reward_factory in METHODS:
        if skip_rl and method_tag == "rl":
            print(f"  [SKIP] {label}")
            continue
 
        print(f"\n  [{label}]  seed={seed}")
        if seed:
            set_global_seeds(seed)
        t0 = time.time()
 
        try:
            if method_tag == "rl":
                metric_summaries = train_and_test_one_policy(reward_factory(), seed=seed)
            elif method_tag == "rl_dcc":
                metric_summaries = train_and_test_one_policy_dcc(reward_factory(), seed=seed)
            elif method_tag == "nsga2_rolling":
                metric_summaries = train_and_test_one_ngsa2(seed=seed)
            elif method_tag == "nsga2_static":
                metric_summaries = train_and_test_one_static_ngsa2(seed=seed)
 
            run_results[label] = extract_metrics(metric_summaries)
            elapsed = time.time() - t0
            sr  = run_results[label]["sharpe_ratio"]
            ret = run_results[label]["final_return"]
            print(f"  Done in {elapsed:.0f}s  |  Return={ret:.4f}  Sharpe={sr:.4f}")
 
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            run_results[label] = None
 
    return run_results
 
 
def compute_aggregate_stats(all_run_results: list) -> dict:
    aggregate = {}
 
    for label, _, _ in METHODS:
        valid = [
            (run_idx, r[label])
            for run_idx, r in enumerate(all_run_results)
            if label in r and r[label] is not None
        ]
 
        if not valid:
            aggregate[label] = None
            continue
 
        best_run_idx, _ = max(valid, key=lambda x: x[1]["sharpe_ratio"])
 
        method_stats = {}
        for metric_key in METRIC_LABELS:
            values = [run[metric_key] for _, run in valid]
            method_stats[metric_key] = {
                "mean":         float(np.mean(values)),
                "std":          float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "best":         float(all_run_results[best_run_idx][label][metric_key]),
                "best_run_idx": best_run_idx,
                "n_valid":      len(valid),
                "all_values":   values,
            }
 
        best_result = all_run_results[best_run_idx][label]
        method_stats["_best_method_label"] = (
            f"{best_result['reward_function']}(GARCH={best_result['garch_volatility']})"
        )
 
        aggregate[label] = method_stats
 
    return aggregate
 
 
def print_summary_table(aggregate: dict, n_runs: int):
    METHOD_COL   = 28
    METRIC_COL   = 28
    metric_keys  = list(METRIC_LABELS.keys())
    total_width  = METHOD_COL + len(metric_keys) * METRIC_COL
 
    print("\n" + "=" * total_width)
    print(f"  EVALUATION RESULTS  —  {n_runs} runs per method")
    print("=" * total_width)
 
    print(f"\n{'Method':<{METHOD_COL}}", end="")
    for k in metric_keys:
        label, arrow = METRIC_LABELS[k]
        print(f"{label+' '+arrow:<{METRIC_COL}}", end="")
    print()
 
    print(f"{'':^{METHOD_COL}}", end="")
    for _ in metric_keys:
        print(f"{'mean ± std  [best]':<{METRIC_COL}}", end="")
    print()
    print("-" * total_width)
 
    for label, _, _ in METHODS:
        stats = aggregate.get(label)
        if stats is None:
            print(f"{label:<{METHOD_COL}}  (all runs failed)")
            continue
 
        n_valid  = next(v for k, v in stats.items() if k != "_best_method_label")["n_valid"]
        row_label = f"{label} (n={n_valid})"
        print(f"{row_label:<{METHOD_COL}}", end="")
 
        for k in metric_keys:
            m    = stats[k]
            cell = f"{m['mean']:.4f}±{m['std']:.4f} [{m['best']:.4f}]"
            print(f"{cell:<{METRIC_COL}}", end="")
        print()
 
    print("=" * total_width + "\n")
 
 
def print_best_run_summary(aggregate: dict):
    print("  BEST RUN PER METHOD  (highest Sharpe)")
    print("  " + "-" * 75)
    for label, _, _ in METHODS:
        stats = aggregate.get(label)
        if stats is None:
            continue
        sr  = stats["sharpe_ratio"]
        ret = stats["final_return"]
        mdd = stats["max_drawdown"]
        best_method = stats.get("_best_method_label", "")
        print(
            f"  {label:<35}  run_idx={sr['best_run_idx']}  "
            f"Sharpe={sr['best']:.4f}  Return={ret['best']:.4f}  MDD={mdd['best']:.4f}  "
            f"Method={best_method}"
        )
    print()
 
 
def save_csv_table(aggregate: dict, output_dir: str):
    path        = os.path.join(output_dir, "results_table.csv")
    metric_keys = list(METRIC_LABELS.keys())
 
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
 
        header = ["Method", "N Runs"]
        for k in metric_keys:
            name = METRIC_LABELS[k][0]
            header += [f"{name} Mean", f"{name} Std", f"{name} Best"]
        header += ["Best Method"]
        writer.writerow(header)
 
        for label, _, _ in METHODS:
            stats = aggregate.get(label)
            if stats is None:
                writer.writerow([label, 0] + [""] * (len(metric_keys) * 3) + [""])
                continue
            n_valid = next(v for k, v in stats.items() if k != "_best_method_label")["n_valid"]
            row = [label, n_valid]
            for k in metric_keys:
                m = stats[k]
                row += [f"{m['mean']:.6f}", f"{m['std']:.6f}", f"{m['best']:.6f}"]
            row += [stats.get("_best_method_label", "")]
            writer.writerow(row)
 
    print(f"CSV table   → {path}")
 
 
def main():
    N_RUNS  = 200
    SKIP_RL = False
    # SEEDS   = list(range(N_RUNS))
    SEEDS  = [None] * N_RUNS
 
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(BASE_DIR, "results", "evaluation", timestamp)
 
    print(f"\n{'='*60}")
    print(f"  Evaluation Framework")
    print(f"  Runs : {N_RUNS}   Seeds: {SEEDS}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}")
 
    all_run_results = []
    total_start     = time.time()
 
    for run_idx, seed in enumerate(SEEDS):
        print(f"\n{'─'*60}")
        print(f"  RUN {run_idx + 1} / {N_RUNS}   (seed={seed})")
        print(f"{'─'*60}")
 
        run_results = run_one_seed(seed, skip_rl=SKIP_RL)
        all_run_results.append(run_results)
 
    print(f"\n  Total time: {(time.time() - total_start) / 60:.1f} min")
 
    aggregate = compute_aggregate_stats(all_run_results)
    print_summary_table(aggregate, N_RUNS)
    print_best_run_summary(aggregate)

    os.makedirs(output_dir, exist_ok=True)
 
    save_csv_table(aggregate, output_dir)
 
 
if __name__ == "__main__":
    main()