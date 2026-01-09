from rl_main import train_and_test_one_policy

def moo(all_policy_metrics): #TO DO: What to do when metrics are almost identical?
    directions = [+1, -1] #Needs to be updated when new metrics are added
    all_non_dominated_policy_metrics = []

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
            all_non_dominated_policy_metrics.append(policy_a)

    return all_non_dominated_policy_metrics

def main():
    all_policy_metrics = []

    for i in range(10):
        all_policy_metrics.append(train_and_test_one_policy())

    print(all_policy_metrics)

    all_non_dominated_policy_metrics = moo(all_policy_metrics)

    print(all_non_dominated_policy_metrics)
       
if __name__ == "__main__":
    main()