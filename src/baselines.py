import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.price_data_reader_writer import load_or_update_price_data
from config.constants import ALL_SYMBOLS_WITH_CASH, GARCH_BURN_IN_DAYS, EVAL_SPLIT
from rl_approach.rl_reward import NGSA2
from rl_approach.rl_env import PortfolioEnv
from test_metrics_calculator import bootstrap_sharpe_test

def run_baseline_env(df, volatilities):
    baseline_weights = np.array([1/len(ALL_SYMBOLS_WITH_CASH)] * len(ALL_SYMBOLS_WITH_CASH))
    env = PortfolioEnv(df, NGSA2(), volatilities)
    obs, _ = env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step(baseline_weights)
    return env.get_metric_summaries()


def get_baseline_returns():
    """Returns equal-weight baseline portfolio returns for test and extra test periods."""
    try:
        prices_df, _, _, test_volatilities, extra_test_volatilities, extra_test_df = load_or_update_price_data()
    except ValueError as error:
        print(f"Fetching Price Data Failed: {error}")
        return None, None

    burn_in_split_index = GARCH_BURN_IN_DAYS + 1 #Prices Length = Returns Length + 1
    eval_split_index = int(len(prices_df[burn_in_split_index:]) * EVAL_SPLIT) + burn_in_split_index

    test_df = prices_df.iloc[eval_split_index:]

    test_summaries       = run_baseline_env(test_df, test_volatilities)
    extra_test_summaries = run_baseline_env(extra_test_df, extra_test_volatilities)

    return (
        np.array(test_summaries["portfolio_returns"].series),
        np.array(extra_test_summaries["portfolio_returns"].series),
    )


def main():
    try:
        prices_df, _, _, test_volatilities, _, _ = load_or_update_price_data()
    except ValueError as error:
        print(f"Fetching Price Data Failed: {error}")
        return

    burn_in_split_index = GARCH_BURN_IN_DAYS + 1 #Prices Length = Returns Length + 1
    eval_split_index = int(len(prices_df[burn_in_split_index:]) * EVAL_SPLIT) + burn_in_split_index

    test_df = prices_df.iloc[eval_split_index:]
    baseline_metric_summaries = run_baseline_env(test_df, test_volatilities)

    baseline_bootstrap_results = bootstrap_sharpe_test(baseline_metric_summaries["portfolio_returns"].series, dates = baseline_metric_summaries["portfolio_dates"].series)

    print(f"Final Portfolio Value = {baseline_metric_summaries["portfolio_value"].aggregate}")
    print(f"Sharpe Ratio = {baseline_metric_summaries["sharpe_ratio"].aggregate}")
    print(f"Max Drawdown = {baseline_metric_summaries["mdd"].aggregate}")
    print(f"Mean HHI = {baseline_metric_summaries["hhi"].aggregate}")
    print(f"Bootstrap p-value = {baseline_bootstrap_results["p_value"]}")
    print(f"Bootstrap Confidence Interval = {baseline_bootstrap_results["ci_95"]}")

    assert baseline_metric_summaries["sharpe_ratio"].aggregate == baseline_bootstrap_results["observed_sharpe"]

if __name__ == "__main__":
    main()