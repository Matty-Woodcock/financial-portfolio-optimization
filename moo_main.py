import matplotlib.pyplot as plt
import numpy as np

from rl_main import train_and_test_one_policy
from rl_reward import PureReturn, LogReturn
from config.constants import ALL_SYMBOLS, ALL_SYMBOLS_WITH_CASH

def extract_aggregates(policy_metric_summaries):
    return [
        policy_metric_summaries["portfolio_value"].aggregate,
        policy_metric_summaries["hhi"].aggregate,
        policy_metric_summaries["mdd"].aggregate
    ]

def moo(all_policy_metrics, directions): #TO DO: What to do when metrics are almost identical?
    all_non_dominated_policy_indices = []

    for i, policy_a in enumerate(all_policy_metrics):
        policy_a_dominated = False
        for j, policy_b in enumerate(all_policy_metrics):
            if i == j:
                continue
            
            policy_b_no_worse = True
            policy_b_strictly_better = False

            for k, d in enumerate(directions):
                if d == 1: #Maximised metrics
                    if policy_a[k] > policy_b[k]:
                        policy_b_no_worse = False
                        break
                    if policy_a[k] < policy_b[k]:
                        policy_b_strictly_better = True
                elif d == -1: #Minimised metrics
                    if policy_a[k] < policy_b[k]:
                        policy_b_no_worse = False
                        break
                    if policy_a[k] > policy_b[k]:
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
    
    reward_functions = [PureReturn(), LogReturn()]

    for i in range(5):
        for reward_function in reward_functions:
            policy_metric_summaries = train_and_test_one_policy(reward_function)
            all_policy_metric_summaries.append(policy_metric_summaries)
            all_policy_metric_aggregates.append(extract_aggregates(policy_metric_summaries))
        
    directions = [
        policy_metric_summaries["portfolio_value"].direction,
        policy_metric_summaries["hhi"].direction,
        policy_metric_summaries["mdd"].direction
    ]

    print(all_policy_metric_aggregates)

    pareto_front_policy_indicies = moo(all_policy_metric_aggregates, directions)

    pareto_front_policy_metric_summaries = [all_policy_metric_summaries[i] for i in pareto_front_policy_indicies]

    print([extract_aggregates(policy_metric_summaries) for policy_metric_summaries in pareto_front_policy_metric_summaries])

    #Experimental Plotting

    cmap = plt.get_cmap("tab10")
    COLOR_MAP = {symbol: cmap(i) for i, symbol in enumerate(ALL_SYMBOLS_WITH_CASH)}

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

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Portfolio Value (Solid)")
        ax2.set_ylabel("HHI (Dotted)")
        plt.title(f"Return = {policy_metric_summaries['portfolio_value'].aggregate}, Reward Function = {policy_metric_summaries['portfolio_value'].reward_function_name}")

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
        ax_w.set_title("Portfolio Weights Over Time")
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
        plt.show()

if __name__ == "__main__":
    main()

#TO DO:
#0. Fix graph colours
#1. Baselines - Done?
#2. Visualisation - Expand
#3. Different Reward Functions
#4. Risk
