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
from config.constants import ALL_SYMBOLS, ALL_SYMBOLS_WITH_CASH, RL_WINDOW_SIZE
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

    #TO DO: Implement with Pandas?

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
    
    # reward_functions = [PureReturn(), LogReturn(), HighHHI(), LowVolatility(),
    #                     PureReturn(volatility_in_observation_space=True),
    #                     LogReturn(volatility_in_observation_space=True),
    #                     HighHHI(volatility_in_observation_space=True),
    #                     LowVolatility(volatility_in_observation_space=True),]
    reward_functions = [PureReturn()]

    for i in range(1):
        for reward_function in reward_functions:
            policy_metric_summaries = train_and_test_one_policy(reward_function)
            all_policy_metric_summaries.append(policy_metric_summaries)
            all_policy_metric_aggregates.append(extract_aggregates(policy_metric_summaries))
            #assert len(policy_metric_summaries["portfolio_value"].series) == 207

    # ngsa2_rolling_metric_summaries = train_and_test_one_ngsa2()
    # all_policy_metric_summaries.append(ngsa2_rolling_metric_summaries)
    # all_policy_metric_aggregates.append(extract_aggregates(ngsa2_rolling_metric_summaries))

    ngsa2_static_metric_summaries = train_and_test_one_static_ngsa2()
    all_policy_metric_summaries.append(ngsa2_static_metric_summaries)
    all_policy_metric_aggregates.append(extract_aggregates(ngsa2_static_metric_summaries))

    # rolling_bootstrap = bootstrap_sharpe_test(
    #     ngsa2_rolling_metric_summaries["portfolio_returns"].series
    # )
    # print(f"\n--- Rolling NSGA-II Bootstrap Results ---")
    # print(f"Observed Sharpe:    {rolling_bootstrap['observed_sharpe']:.3f}")
    # print(f"95% CI:             ({rolling_bootstrap['ci_95'][0]:.3f}, {rolling_bootstrap['ci_95'][1]:.3f})")
    # print(f"P-value (H0: SR≤0): {rolling_bootstrap['p_value']:.3f}")

    # static_bootstrap = bootstrap_sharpe_test(
    #     ngsa2_static_metric_summaries["portfolio_returns"].series
    # )
    # print(f"\n--- Static NSGA-II Bootstrap Results ---")
    # print(f"Observed Sharpe:    {static_bootstrap['observed_sharpe']:.3f}")
    # print(f"95% CI:             ({static_bootstrap['ci_95'][0]:.3f}, {static_bootstrap['ci_95'][1]:.3f})")
    # print(f"P-value (H0: SR≤0): {static_bootstrap['p_value']:.3f}")

    # print("\nLength Rolling EA:", len(ngsa2_rolling_metric_summaries["portfolio_value"].series))
    # print(f"Length Static EA: {len(ngsa2_static_metric_summaries["portfolio_value"].series)}\n")
    
    directions = [
        all_policy_metric_summaries[0]["portfolio_value"].direction,
        all_policy_metric_summaries[0]["hhi"].direction,
        all_policy_metric_summaries[0]["mdd"].direction
    ]
    print(all_policy_metric_aggregates)

    pareto_front_policy_indicies = moo(all_policy_metric_aggregates, directions)

    pareto_front_policy_metric_summaries = [all_policy_metric_summaries[i] for i in pareto_front_policy_indicies]

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
    #all_hhi = [p["hhi"].series for p in pareto_front_policy_metric_summaries[:-1]] #TO DO: Fix
    all_hhi = [p["hhi"].series for p in all_policy_metric_summaries[:-1]]

    pv_min = min(min(pv) for pv in all_portfolio_values)
    pv_max = max(max(pv) for pv in all_portfolio_values)
    hhi_max = max(max(h) for h in all_hhi)
    vol_max = max(max(v) for v in all_rolling_volatilities)

    for policy_metric_summaries in pareto_front_policy_metric_summaries:
        policy_bootstrap_results = bootstrap_sharpe_test(policy_metric_summaries["portfolio_returns"].series)

        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(12, 10),
            sharex=True
        )

        ax1 = axes[0]
        ax2 = ax1.twinx() #Second y-axis for HHI

        n_trading_days = len(policy_metric_summaries["portfolio_value"].series)
        trading_days = np.arange(1, n_trading_days + 1) #TO DO: Is this necessary?

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
        print(len(policy_metric_summaries["portfolio_value"].series), len(policy_metric_summaries["hhi"].series))
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
            f"observed sharpe: {policy_bootstrap_results["observed_sharpe"]:.4f}\n" #TO DO: REMOVE
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

        prices = policy_metric_summaries["ticker_prices"].series[RL_WINDOW_SIZE - 1:] #Prices in initial observation window are discarded 39:
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
#1. Satistical significance
#2. Don't use MDD in NGSA-2?
#2. Remove XOM & BA?
#4. DCC-GARCH?

#[np.float64(1.1005400115336867), np.float64(1.0653347873779786), np.float64(1.0739545460619913), np.float64(1.0893033098977247), np.float64(1.087608475800012), np.float64(1.0867396581242237), np.float64(1.1137366773873456), np.float64(1.071185103310227), np.float64(0.9973951386100808), np.float64(0.967510795589207), np.float64(0.96511207172244), np.float64(0.9629288624415098), np.float64(0.9879293326827104), np.float64(0.9889782013844942), np.float64(0.9782271599951061), np.float64(1.00180502439772), np.float64(1.0006796758967038), np.float64(1.0460907500982408), np.float64(1.0542403944296377), np.float64(1.0570012477036308), np.float64(1.0385678373258074), np.float64(1.0461187121885012), np.float64(1.0318316068170306), np.float64(1.0407772040628436), np.float64(1.0318392920137198), np.float64(1.023139513306916), np.float64(1.0093710070897748), np.float64(1.017803161344673), np.float64(1.0421147941874171), np.float64(0.999743161137668), np.float64(0.9971694826374349), np.float64(1.0082342417196657), np.float64(0.9754085843470889), np.float64(1.002819544812224), np.float64(1.0074715721587142), np.float64(1.0261210845123192), np.float64(1.0377681859449484), np.float64(1.0185409058255461), np.float64(1.0101782522601495), np.float64(1.0185432753359942), np.float64(1.0133671483438522), np.float64(1.0218379855402304), np.float64(1.0228612794269474), np.float64(1.0386959208594313), np.float64(1.03741884120459), np.float64(1.0222463955059828), np.float64(1.0256754641190216), np.float64(1.0218262307027572), np.float64(1.00408750856496), np.float64(1.0072783038773476), np.float64(0.9768639574467901), np.float64(0.9671401691593584), np.float64(0.9836342995522797), np.float64(0.9773416393639615), np.float64(0.9799559053499967), np.float64(0.9690281211689322), np.float64(0.9596805658254094), np.float64(0.9727640095442435), np.float64(0.9724196546617683), np.float64(0.9882299759828763), np.float64(0.9935651553849862), np.float64(0.9960581015150363), np.float64(1.0008856454025588), np.float64(1.019301571804504), np.float64(1.0193604997963421), np.float64(1.000225255139941), np.float64(0.9945078233910798), np.float64(0.9955265381786569), np.float64(0.9848221646578842), np.float64(0.9938452314256604), np.float64(0.9925346739662014), np.float64(0.9675577930886489), np.float64(0.9795248462383346), np.float64(0.9795966743338745), np.float64(0.9692318646446427), np.float64(0.9528668624993331), np.float64(0.9750280634220941), np.float64(0.9763006492256561), np.float64(0.9606611725670379), np.float64(0.947477976790869), np.float64(0.9308622707634762), np.float64(0.9296096083389758), np.float64(0.9312024433293552), np.float64(0.9502384848560537), np.float64(0.9572428319636505), np.float64(0.960530083995319), np.float64(0.9514315897514359), np.float64(0.9746613060432667), np.float64(0.9900453745127463), np.float64(0.9862548047476646), np.float64(0.9916951320765333), np.float64(1.006618015706152), np.float64(1.0133745580622842), np.float64(1.0136879543576325), np.float64(1.02551789406247), np.float64(1.0195993348817136), np.float64(1.0102021679165514)]

#Liagkouras and Metaxiotis 2013

#Fallback
#0.65, 0.75, 20000, no weights in observation window, 0.05 random