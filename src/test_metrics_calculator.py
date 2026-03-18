import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  
import matplotlib.pyplot as plt

from dataclasses import dataclass
from arch.bootstrap import CircularBlockBootstrap

from data.rf_rate_reader import get_daily_rf, RF_DAILY_SERIES
from config.constants import DATA_START, DATA_END, EVAL_SPLIT

@dataclass
class MetricSummary:
    name: str
    reward_function_name: str
    direction: int
    series: list[float]
    aggregate: float

#Herfindahl–Hirschman Index (HHI)
def calculate_hhi_metric(weights_history):
    hhi_history = []
    for daily_weights in weights_history: 
        hhi_history.append(np.sum(np.square(daily_weights)))
    return hhi_history, np.mean(hhi_history)

#Maximum Drawdown (MDD)
def calculate_mdd_metric(portfolio_value_history):
    mdd_history = []
    peak_porfolio_value = 1
    for daily_portfolio_value in portfolio_value_history:
        peak_porfolio_value = max(peak_porfolio_value, daily_portfolio_value)
        mdd_history.append((daily_portfolio_value - peak_porfolio_value) / peak_porfolio_value)
    return mdd_history, min(mdd_history)

def calculate_sharpe_ratio(portfolio_return_history, dates = None):
    returns = np.array(portfolio_return_history)

    if dates is not None and len(dates) == len(returns): #TO DO: Needs fixing
        rf_daily = np.array([get_daily_rf(date) for date in dates])
    else:
        test_start = DATA_START + (DATA_END - DATA_START) * EVAL_SPLIT
        rf_daily = RF_DAILY_SERIES[test_start:DATA_END].mean()

    excess = returns - rf_daily
    return (excess.mean() / excess.std(ddof=1)) * np.sqrt(252)

#TO DO: Review AI
def bootstrap_sharpe_test(portfolio_return_history, n_bootstraps=1000, block_size=10):
    """
    Block bootstrap test of H0: Sharpe ratio <= 0
    Returns the p-value (one-tailed) and a 95% confidence interval.
    Block size of ~sqrt(T) is the standard rule of thumb.
    """
    returns = np.array(portfolio_return_history)

    bs = CircularBlockBootstrap(block_size, returns)
    bootstrap_sharpes = bs.apply(calculate_sharpe_ratio, n_bootstraps)

    observed_sharpe = calculate_sharpe_ratio(returns) #TO DO: Needs dates
    p_value = float(np.mean(bootstrap_sharpes <= 0))  #H0: Sharpe <= 0
    ci_lower = float(np.percentile(bootstrap_sharpes, 2.5))
    ci_upper = float(np.percentile(bootstrap_sharpes, 97.5))

    return {
        "observed_sharpe": observed_sharpe,
        "p_value": p_value,
        "ci_95": (ci_lower, ci_upper),
        "bootstrap_sharpes": bootstrap_sharpes
    }

