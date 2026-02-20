import matplotlib.pyplot as plt
import numpy as np
import os

from datetime import datetime

from rl_main import train_and_test_one_policy
from ngsa2_main import train_and_test_one_ngsa2
from rl_reward import PureReturn, LogReturn, HighHHI
from config.constants import ALL_SYMBOLS, ALL_SYMBOLS_WITH_CASH

def extract_aggregates(policy_metric_summaries):
    return [
        policy_metric_summaries["portfolio_value"].aggregate,
        policy_metric_summaries["hhi"].aggregate,
        policy_metric_summaries["mdd"].aggregate
    ]
    # return [
    #         round(policy_metric_summaries["portfolio_value"].aggregate, 4),
    #         round(policy_metric_summaries["hhi"].aggregate, 4),
    #         round(policy_metric_summaries["mdd"].aggregate, 4)
    #     ]


def rolling_volatility_calculator(portfolio_values, window_size=20):
    portfolio_values = np.array(portfolio_values)
    portfolio_returns = portfolio_values[1:] / portfolio_values[:-1] - 1

    rolling_volatility = np.full(len(portfolio_values), np.nan)
    for i in range(window_size, len(portfolio_values)):
        rolling_volatility[i] = np.std(portfolio_returns[i - window_size:i], ddof=1)
    
    rolling_volatility_annual = np.array(rolling_volatility) * np.sqrt(252)
    return rolling_volatility_annual

    #TO DO: Implement with Pandas?

def moo(all_policy_metrics, directions): #TO DO: What to do when metrics are almost identical?
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
    
    #reward_functions = [PureReturn(), LogReturn(), HighHHI(),
                        # PureReturn(volatility_in_observation_space=True),
                        # LogReturn(volatility_in_observation_space=True),
                        # HighHHI(volatility_in_observation_space=True),]
    reward_functions = []

    for i in range(1):
        for reward_function in reward_functions:
            policy_metric_summaries = train_and_test_one_policy(reward_function)
            all_policy_metric_summaries.append(policy_metric_summaries)
            all_policy_metric_aggregates.append(extract_aggregates(policy_metric_summaries))
        ngsa2_metric_summaries = train_and_test_one_ngsa2(PureReturn())
        all_policy_metric_summaries.append(ngsa2_metric_summaries)
        # print(ngsa2_metric_summaries)
        # input("X")
        all_policy_metric_aggregates.append(extract_aggregates(ngsa2_metric_summaries))
        
    directions = [
        all_policy_metric_summaries[0]["portfolio_value"].direction,
        all_policy_metric_summaries[0]["hhi"].direction,
        all_policy_metric_summaries[0]["mdd"].direction
    ]

    print(all_policy_metric_aggregates)

    pareto_front_policy_indicies = moo(all_policy_metric_aggregates, directions)

    pareto_front_policy_metric_summaries = [all_policy_metric_summaries[i] for i in pareto_front_policy_indicies]

    print([extract_aggregates(policy_metric_summaries) for policy_metric_summaries in pareto_front_policy_metric_summaries])

    #Experimental Plotting

    cmap = plt.get_cmap("tab20")
    COLOR_MAP = {symbol: cmap(i) for i, symbol in enumerate(ALL_SYMBOLS_WITH_CASH)}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_DIR = os.path.join("results", timestamp)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pareto_front_counter = 0

    for policy_metric_summaries in pareto_front_policy_metric_summaries:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(12, 10),
            sharex=True
        )

        ax1 = axes[0]
        ax2 = ax1.twinx() #Second y-axis for HHI

        ax1.plot(
            policy_metric_summaries["portfolio_value"].series,
            label="Portfolio Value",
            alpha=0.8
        )

        ax2.plot(
            policy_metric_summaries["hhi"].series,
            linestyle=":",
            label="HHI",
            alpha=0.8
        )
        rolling_volatility = rolling_volatility_calculator(policy_metric_summaries["portfolio_value"].series)
        #print(len(rolling_volatility), len(policy_metric_summaries["portfolio_value"].series), len(policy_metric_summaries["hhi"].series))
        # print(rolling_volatility)
        assert len(rolling_volatility) == len(policy_metric_summaries["portfolio_value"].series) == len(policy_metric_summaries["hhi"].series)
        ax2.plot(
            rolling_volatility,
            linestyle="--",
            label="Rolling Volatility",
            alpha=0.8,
            color="r"
        )

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Portfolio Value (Blue)")
        ax2.set_ylabel("HHI (Dotted) & Rolling Annualised Volatility (Red)")

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

        ax_w.set_ylabel("Weight")
        ax_w.set_xlabel("Days")
        #TO DO: Are signficant figures possible below?
        ax_w.set_title(f"Portfolio Weights Over Time (Total Transaction Cost Percentage Relative To Initial Capital = {policy_metric_summaries['transaction_cost_percentage'].aggregate:.3f}%)")
        ax_w.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=len(ALL_SYMBOLS_WITH_CASH)
        )

        #Ticker Prices

        prices = policy_metric_summaries["ticker_prices"].series[39:] #Prices in initial observation window are discarded
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

if __name__ == "__main__":
    main()

#TO DO
#1. Speed up
#2. Seperate MOO
#3. DCC-GARCH?

