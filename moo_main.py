import matplotlib.pyplot as plt

from rl_main import train_and_test_one_policy

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

    for i in range(1):
        policy_metric_summaries = train_and_test_one_policy()
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

    for policy_metric_summaries in pareto_front_policy_metric_summaries:
        fig, ax1 = plt.subplots()

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
        plt.title(f"Return = {policy_metric_summaries['portfolio_value'].aggregate}")

        plt.show()

if __name__ == "__main__":
    main()

#TO DO:
#1. Baselines - Done?
#2. Visualisation - Expand
#3. Different Reward Functions
#4. Risk
