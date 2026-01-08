from rl_main import train_and_test_one_policy

def main():
    test_metrics_per_policy = []

    for i in range(1):
        test_metrics_per_policy.append(train_and_test_one_policy())

    print(sorted(test_metrics_per_policy))

if __name__ == "__main__":
    main()