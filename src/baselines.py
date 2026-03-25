import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.price_data_reader_writer import load_or_update_price_data
from config.constants import ALL_SYMBOLS_WITH_CASH, GARCH_BURN_IN_DAYS, TRAIN_SPLIT, EVAL_SPLIT
from rl_approach.rl_reward import NGSA2
from rl_approach.rl_env import PortfolioEnv
from test_metrics_calculator import bootstrap_sharpe_test

def main():
    try:
        prices_df, _, _, test_volatilities, _ = load_or_update_price_data()
    except ValueError as error:
        print(f"Fetching Price Data Failed: {error}")
        return

    burn_in_split_index = GARCH_BURN_IN_DAYS + 1 #Prices Length = Returns Length + 1
    train_split_index = int(len(prices_df[burn_in_split_index:]) * TRAIN_SPLIT) + burn_in_split_index
    eval_split_index = int(len(prices_df[burn_in_split_index:]) * EVAL_SPLIT) + burn_in_split_index
    
    test_df = prices_df.iloc[eval_split_index:]
    test_env = PortfolioEnv(test_df, NGSA2(), test_volatilities)

    baseline_weights = np.array([1/len(ALL_SYMBOLS_WITH_CASH)] * len(ALL_SYMBOLS_WITH_CASH))
    
    obs, _ = test_env.reset()
    done = False

    while not done:
        _, _, done, _, _ = test_env.step(baseline_weights) #TO DO: In some cases, the ratios wont't be preserved

    baseline_metric_summaries = test_env.get_metric_summaries()

    baseline_bootstrap_results = bootstrap_sharpe_test(baseline_metric_summaries["portfolio_returns"].series)

    print(f"Final Portfolio Value = {baseline_metric_summaries["portfolio_value"].aggregate}")
    print(f"Sharpe Ratio = {baseline_metric_summaries["sharpe_ratio"].aggregate}")
    print(f"Max Drawdown = {baseline_metric_summaries["mdd"].aggregate}")
    print(f"Mean HHI = {baseline_metric_summaries["hhi"].aggregate}")
    print(f"Bootstrap p-value = {baseline_bootstrap_results["p_value"]}")
    print(f"Bootstrap Confidence Interval = {baseline_bootstrap_results["ci_95"]}")

    #TO DO: Observed sharped incorrectly 0.6% lower
    #print(f"Bootstrap Observed Sharpe = {baseline_bootstrap_results["observed_sharpe"]}")

if __name__ == "__main__":
    main()