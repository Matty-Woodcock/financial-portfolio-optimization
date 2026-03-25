import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  
import matplotlib.pyplot as plt

from dataclasses import dataclass
from arch.bootstrap import CircularBlockBootstrap

from data.rf_rate_reader import get_daily_rf, RF_DAILY_SERIES
from config.constants import DATA_START, DATA_END, EVAL_SPLIT, EXTRA_TEST_END

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


def calculate_sharpe_ratio(portfolio_return_history, dates=None, end_date=DATA_END):
    returns = np.array(portfolio_return_history)

    if dates is not None and len(dates) == len(returns):
        rf_daily = np.array([get_daily_rf(date) for date in dates])
    else:
        if end_date == DATA_END:
            test_start = DATA_START + (DATA_END - DATA_START) * EVAL_SPLIT
        else:
            test_start = DATA_END
        rf_daily = RF_DAILY_SERIES[test_start:end_date].mean()

    excess = returns - rf_daily
    return (excess.mean() / excess.std(ddof=1)) * np.sqrt(252)

_baseline_cache = None

def bootstrap_sharpe_baseline_test(model_returns, dates=None, n_bootstraps=1000, block_size=10, end_date=DATA_END):
    """
    Tests H0: Sharpe(model) <= Sharpe(baseline).
    Baseline is the equal-weight portfolio. Uses end_date to select test vs extra test baseline.
    Uses paired circular block bootstrap — same block indices applied to both series.
    p_value = proportion of bootstrap samples where Sharpe(model) - Sharpe(baseline) <= 0.
    """
    global _baseline_cache
    if _baseline_cache is None:
        from baselines import get_baseline_returns #Avoids circular import
        test_returns, extra_test_returns = get_baseline_returns()
        _baseline_cache = {DATA_END: test_returns, EXTRA_TEST_END: extra_test_returns}

    model_returns = np.array(model_returns)
    baseline_returns = _baseline_cache[end_date]
    assert len(model_returns) == len(baseline_returns)

    observed_diff = (calculate_sharpe_ratio(model_returns, dates=dates, end_date=end_date) -
                     calculate_sharpe_ratio(baseline_returns, dates=dates, end_date=end_date))

    bs = CircularBlockBootstrap(block_size, model_returns, baseline_returns)
    bootstrap_diffs = bs.apply(
        lambda m, b: calculate_sharpe_ratio(m, end_date=end_date) - calculate_sharpe_ratio(b, end_date=end_date),
        n_bootstraps
    )

    p_value  = float(np.mean(bootstrap_diffs <= 0))
    ci_lower = float(np.percentile(bootstrap_diffs, 2.5))
    ci_upper = float(np.percentile(bootstrap_diffs, 97.5))

    return {
        "observed_diff": float(observed_diff),
        "p_value":       p_value,
        "ci_95":         (ci_lower, ci_upper),
    }

#TO DO: Review AI
def bootstrap_sharpe_test(portfolio_return_history, dates=None, n_bootstraps=1000, block_size=10, end_date=DATA_END):
    returns = np.array(portfolio_return_history)

    bs = CircularBlockBootstrap(block_size, returns)
    bootstrap_sharpes = bs.apply(
        lambda x: calculate_sharpe_ratio(x, end_date=end_date),
        n_bootstraps
    )

    observed_sharpe = calculate_sharpe_ratio(returns, dates=dates, end_date=end_date)
    p_value = float(np.mean(bootstrap_sharpes <= 0))
    ci_lower = float(np.percentile(bootstrap_sharpes, 2.5))
    ci_upper = float(np.percentile(bootstrap_sharpes, 97.5))

    return {
        "observed_sharpe": observed_sharpe,
        "p_value": p_value,
        "ci_95": (ci_lower, ci_upper),
        "bootstrap_sharpes": bootstrap_sharpes
    }
