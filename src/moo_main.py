import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import os

from datetime import datetime

from rl_approach.rl_main import train_and_test_one_policy
from ea_approach.ngsa2_main import train_and_test_one_ngsa2
from ea_approach.ngsa2_static import train_and_test_one_static_ngsa2
from rl_approach.rl_reward import PureReturn, LogReturn, HighHHI, LowVolatility
from config.constants import ALL_SYMBOLS, ALL_SYMBOLS_WITH_CASH, RL_WINDOW_SIZE, EXTRA_TEST_END
from test_metrics_calculator import bootstrap_sharpe_test

def extract_aggregates(policy_metric_summaries):
    return [
        policy_metric_summaries["portfolio_value"].aggregate,
        policy_metric_summaries["hhi"].aggregate,
        policy_metric_summaries["mdd"].aggregate
    ]

def rolling_volatility_calculator(portfolio_values, window_size=20):
    portfolio_values = np.array(portfolio_values)
    portfolio_returns = portfolio_values[1:] / portfolio_values[:-1] - 1

    rolling_volatility = np.full(len(portfolio_values), np.nan)
    for i in range(window_size, len(portfolio_values)):
        rolling_volatility[i] = np.std(portfolio_returns[i - window_size:i], ddof=1)
    
    rolling_volatility_annual = np.array(rolling_volatility) * np.sqrt(252)
    return rolling_volatility_annual

def moo(all_policy_metrics, directions):
    all_non_dominated_policy_indices = []
    epsilion = 0.005

    for i, policy_a in enumerate(all_policy_metrics):
        policy_a_dominated = False
        for j, policy_b in enumerate(all_policy_metrics):
            if i == j:
                continue
            
            policy_b_no_worse = True
            policy_b_strictly_better = False

            for k, d in enumerate(directions):
                if d == 1: #Maximised metrics
                    if policy_a[k] > policy_b[k] + epsilion:
                        policy_b_no_worse = False
                        break
                    if policy_a[k] + epsilion < policy_b[k]:
                        policy_b_strictly_better = True
                elif d == -1: #Minimised metrics
                    if policy_a[k] < policy_b[k] - epsilion:
                        policy_b_no_worse = False
                        break
                    if policy_a[k] - epsilion > policy_b[k]:
                        policy_b_strictly_better = True

            if policy_b_no_worse and policy_b_strictly_better:
                policy_a_dominated = True
                print(f"Policy {i} Dominated By Policy {j}")     
                break

        if not policy_a_dominated:
            all_non_dominated_policy_indices.append(i)

    return all_non_dominated_policy_indices

def main():
    all_policy_metric_summaries = []
    all_policy_metric_aggregates = []
    all_policy_extra_test_summaries = []

    reward_functions = [PureReturn(), LogReturn(), HighHHI(), LowVolatility(),
                        PureReturn(volatility_in_observation_space=True),
                        LogReturn(volatility_in_observation_space=True),
                        HighHHI(volatility_in_observation_space=True),
                        LowVolatility(volatility_in_observation_space=True),]

    for i in range(1):
        for reward_function in reward_functions:
            result = train_and_test_one_policy(reward_function)
            all_policy_metric_summaries.append(result["test"])
            all_policy_metric_aggregates.append(extract_aggregates(result["test"]))
            all_policy_extra_test_summaries.append(result["extra_test"])

    #Evolutionary Algorithms

    ngsa2_rolling_metric_summaries = train_and_test_one_ngsa2()
    all_policy_metric_summaries.append(ngsa2_rolling_metric_summaries["test"])
    all_policy_metric_aggregates.append(extract_aggregates(ngsa2_rolling_metric_summaries["test"]))
    all_policy_extra_test_summaries.append(ngsa2_rolling_metric_summaries["extra_test"])

    ngsa2_static_metric_summaries = train_and_test_one_static_ngsa2()
    all_policy_metric_summaries.append(ngsa2_static_metric_summaries["test"])
    all_policy_metric_aggregates.append(extract_aggregates(ngsa2_static_metric_summaries["test"]))
    all_policy_extra_test_summaries.append(ngsa2_static_metric_summaries["extra_test"])
    
    directions = [
        all_policy_metric_summaries[0]["portfolio_value"].direction,
        all_policy_metric_summaries[0]["hhi"].direction,
        all_policy_metric_summaries[0]["mdd"].direction
    ]
    print(all_policy_metric_aggregates)

    pareto_front_policy_indicies = moo(all_policy_metric_aggregates, directions)

    pareto_front_policy_metric_summaries = [all_policy_metric_summaries[i] for i in pareto_front_policy_indicies]
    pareto_front_extra_test_metric_summaries = [extra for extra in [all_policy_extra_test_summaries[i] for i in pareto_front_policy_indicies] if extra is not None]
    print([extract_aggregates(policy_metric_summaries) for policy_metric_summaries in pareto_front_policy_metric_summaries])

    #Plotting Results Graphs

    cmap = plt.get_cmap("tab20")
    COLOR_MAP = {symbol: cmap(i) for i, symbol in enumerate(ALL_SYMBOLS_WITH_CASH)}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_DIR = os.path.join(BASE_DIR, "results", "graphs", timestamp)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pareto_front_counter = 0
        
    all_portfolio_values = [p["portfolio_value"].series for p in pareto_front_policy_metric_summaries]
    all_rolling_volatilities = [rolling_volatility_calculator(p["portfolio_value"].series) for p in pareto_front_policy_metric_summaries]
    all_hhi = [p["hhi"].series for p in pareto_front_policy_metric_summaries]

    pv_min = min(min(pv) for pv in all_portfolio_values)
    pv_max = max(max(pv) for pv in all_portfolio_values)
    hhi_max = max(max(h) for h in all_hhi)
    vol_max = max(float(np.nanmax(v)) for v in all_rolling_volatilities)

    for policy_metric_summaries in pareto_front_policy_metric_summaries:
        policy_bootstrap_results = bootstrap_sharpe_test(policy_metric_summaries["portfolio_returns"].series, dates = policy_metric_summaries["portfolio_dates"].series)

        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(12, 10),
            sharex=True
        )

        ax1 = axes[0]
        ax2 = ax1.twinx() #Second y-axis for HHI

        n_trading_days = len(policy_metric_summaries["portfolio_value"].series)
        trading_days = np.arange(1, n_trading_days + 1)

        ax1.plot(
            trading_days,
            policy_metric_summaries["portfolio_value"].series,
            label="Portfolio Value",
            alpha=0.8
        )

        ax2.plot(
            trading_days,
            policy_metric_summaries["hhi"].series,
            linestyle=":",
            label="HHI",
            alpha=0.8
        )
        rolling_volatility = all_rolling_volatilities[pareto_front_counter]
        assert len(rolling_volatility) == len(policy_metric_summaries["portfolio_value"].series) == len(policy_metric_summaries["hhi"].series)
        ax2.plot(
            trading_days,
            rolling_volatility,
            linestyle="--",
            label="Rolling Volatility",
            alpha=0.8,
            color="r"
        )

        ax1.xaxis.set_tick_params(labelbottom=True)
        ax1.set_xlabel("Days")
        ax1.set_ylabel("Portfolio Value (Blue)")
        ax2.set_ylabel("HHI (Dotted) & Rolling Annualised Volatility (Red)")

        ax1.set_ylim([pv_min * 0.95, pv_max * 1.05])

        p_value = policy_bootstrap_results["p_value"]
        ci_low, ci_high = policy_bootstrap_results["ci_95"]

        #Colour the p-value green/orange/red depending on significance
        if p_value < 0.05:
            p_color = "green"
        elif p_value < 0.10:
            p_color = "orange"
        else:
            p_color = "red"

        stats_text = (
            f"p-value: {p_value:.3f}\n"
            f"95% CI: ({ci_low:.2f}, {ci_high:.2f})\n"
            f"H₀: Sharpe Ratio ≤ 0"
        )

        ax1.text(
            0.015, 0.95, #Top-left of axes
            stats_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=p_color, alpha=0.15)
        )

        ax2.set_ylim([0.0, max(hhi_max, vol_max) * 1.03])

        rounded_return = round(policy_metric_summaries['portfolio_value'].aggregate, 3)
        rounded_sharpe_ratio = round(policy_metric_summaries['sharpe_ratio'].aggregate, 3)
        plt.title(f"Return = {rounded_return}, Sharpe Ratio = {rounded_sharpe_ratio}, Reward Function = {policy_metric_summaries['portfolio_value'].reward_function_name}, GARCH Volatility = {policy_metric_summaries['garch_volatility'].name}")
 
        #Portfolio Weights

        weights = np.vstack(policy_metric_summaries["portfolio_weights"].series)
        t = np.arange(weights.shape[0])

        assert weights.shape[1] == len(ALL_SYMBOLS_WITH_CASH)

        ax_w = axes[1]
        ax_w.stackplot(
            t,
            weights.T,
            colors=[COLOR_MAP[symbol] for symbol in ALL_SYMBOLS_WITH_CASH],
            labels=ALL_SYMBOLS_WITH_CASH
        )

        ax_w.xaxis.set_tick_params(labelbottom=True)
        ax_w.set_ylabel("Weight")
        ax_w.set_xlabel("Days")

        ax_w.set_title(f"Portfolio Weights Over Time (Total Transaction Cost Percentage Relative To Initial Capital = {policy_metric_summaries['transaction_cost_percentage'].aggregate:.3f}%)")
        ax_w.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=len(ALL_SYMBOLS_WITH_CASH)
        )

        #Ticker Prices

        prices = policy_metric_summaries["ticker_prices"].series[RL_WINDOW_SIZE - 1:] #Prices in initial observation window are discarded
        prices_norm = prices / prices[0]

        ax_p = axes[2]

        for i, symbol in enumerate(ALL_SYMBOLS):
            ax_p.plot(prices_norm[:, i], color=COLOR_MAP[symbol], label=symbol)

        ax_p.set_ylabel("Normalised Price")
        ax_p.set_xlabel("Days")
        ax_p.set_title("Ticker Prices (Normalised)")

        ax_p.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=len(ALL_SYMBOLS)
        )

        plt.tight_layout()

        filename = f"sharpe_{policy_metric_summaries['sharpe_ratio'].aggregate:.2f}_{pareto_front_counter}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.show()

        pareto_front_counter += 1

    #Extra Test Graphs

    if pareto_front_extra_test_metric_summaries:
        all_extra_portfolio_values     = [p["portfolio_value"].series for p in pareto_front_extra_test_metric_summaries]
        all_extra_rolling_volatilities = [rolling_volatility_calculator(p["portfolio_value"].series) for p in pareto_front_extra_test_metric_summaries]
        all_extra_hhi                  = [p["hhi"].series for p in pareto_front_extra_test_metric_summaries]

        extra_pv_min  = min(min(pv) for pv in all_extra_portfolio_values)
        extra_pv_max  = max(max(pv) for pv in all_extra_portfolio_values)
        extra_hhi_max = max(max(h) for h in all_extra_hhi)
        extra_vol_max = max(float(np.nanmax(v)) for v in all_extra_rolling_volatilities)

        extra_counter = 0

        for policy_metric_summaries in pareto_front_extra_test_metric_summaries:
            policy_bootstrap_results = bootstrap_sharpe_test(policy_metric_summaries["portfolio_returns"].series, dates = policy_metric_summaries["portfolio_dates"].series, end_date = EXTRA_TEST_END)

            fig, axes = plt.subplots(
                nrows=3,
                ncols=1,
                figsize=(12, 10),
                sharex=True
            )

            ax1 = axes[0]
            ax2 = ax1.twinx()

            n_trading_days = len(policy_metric_summaries["portfolio_value"].series)
            trading_days = np.arange(1, n_trading_days + 1)

            ax1.plot(
                trading_days,
                policy_metric_summaries["portfolio_value"].series,
                label="Portfolio Value",
                alpha=0.8
            )

            ax2.plot(
                trading_days,
                policy_metric_summaries["hhi"].series,
                linestyle=":",
                label="HHI",
                alpha=0.8
            )
            rolling_volatility = all_extra_rolling_volatilities[extra_counter]
            assert len(rolling_volatility) == len(policy_metric_summaries["portfolio_value"].series) == len(policy_metric_summaries["hhi"].series)
            ax2.plot(
                trading_days,
                rolling_volatility,
                linestyle="--",
                label="Rolling Volatility",
                alpha=0.8,
                color="r"
            )

            ax1.xaxis.set_tick_params(labelbottom=True)
            ax1.set_xlabel("Days")
            ax1.set_ylabel("Portfolio Value (Blue)")
            ax2.set_ylabel("HHI (Dotted) & Rolling Annualised Volatility (Red)")

            ax1.set_ylim([extra_pv_min * 0.95, extra_pv_max * 1.05])

            p_value = policy_bootstrap_results["p_value"]
            ci_low, ci_high = policy_bootstrap_results["ci_95"]

            if p_value < 0.05:
                p_color = "green"
            elif p_value < 0.10:
                p_color = "orange"
            else:
                p_color = "red"

            stats_text = (
                f"p-value: {p_value:.3f}\n"
                f"95% CI: ({ci_low:.2f}, {ci_high:.2f})\n"
                f"H₀: Sharpe Ratio ≤ 0"
            )

            ax1.text(
                0.015, 0.95,
                stats_text,
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=p_color, alpha=0.15)
            )

            ax2.set_ylim([0.0, max(extra_hhi_max, extra_vol_max) * 1.03])

            rounded_return = round(policy_metric_summaries['portfolio_value'].aggregate, 3)
            rounded_sharpe_ratio = round(policy_metric_summaries['sharpe_ratio'].aggregate, 3)
            plt.title(f"[Extra Test] Return = {rounded_return}, Sharpe Ratio = {rounded_sharpe_ratio}, Reward Function = {policy_metric_summaries['portfolio_value'].reward_function_name}, GARCH Volatility = {policy_metric_summaries['garch_volatility'].name}")

            #Portfolio Weights

            weights = np.vstack(policy_metric_summaries["portfolio_weights"].series)
            t = np.arange(weights.shape[0])

            assert weights.shape[1] == len(ALL_SYMBOLS_WITH_CASH)

            ax_w = axes[1]
            ax_w.stackplot(
                t,
                weights.T,
                colors=[COLOR_MAP[symbol] for symbol in ALL_SYMBOLS_WITH_CASH],
                labels=ALL_SYMBOLS_WITH_CASH
            )

            ax_w.xaxis.set_tick_params(labelbottom=True)
            ax_w.set_ylabel("Weight")
            ax_w.set_xlabel("Days")

            ax_w.set_title(f"Portfolio Weights Over Time (Total Transaction Cost Percentage Relative To Initial Capital = {policy_metric_summaries['transaction_cost_percentage'].aggregate:.3f}%)")
            ax_w.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                ncol=len(ALL_SYMBOLS_WITH_CASH)
            )

            #Ticker Prices

            prices = policy_metric_summaries["ticker_prices"].series[RL_WINDOW_SIZE - 1:]
            prices_norm = prices / prices[0]

            ax_p = axes[2]

            for i, symbol in enumerate(ALL_SYMBOLS):
                ax_p.plot(prices_norm[:, i], color=COLOR_MAP[symbol], label=symbol)

            ax_p.set_ylabel("Normalised Price")
            ax_p.set_xlabel("Days")
            ax_p.set_title("Ticker Prices (Normalised)")

            ax_p.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                ncol=len(ALL_SYMBOLS)
            )

            plt.tight_layout()

            filename = f"extra_test_sharpe_{policy_metric_summaries['sharpe_ratio'].aggregate:.2f}_{extra_counter}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)

            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.show()

            extra_counter += 1

if __name__ == "__main__":
    main()