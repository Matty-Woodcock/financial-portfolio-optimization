import matplotlib.pyplot as plt

def plot_pareto_front(objective_one, objective_two, best_index):
    plt.figure(figsize=(8,6))
    plt.scatter(objective_one, objective_two, c='blue', label='Pareto Solutions')
    plt.scatter(objective_one[best_index], objective_two[best_index], 
                c='red', marker='*', s=200, label='Chosen (Elbow)', zorder=5)
    plt.xlabel("Return")
    plt.ylabel("Max Drawdown")
    plt.title("Pareto Front")
    plt.grid(True)

    plt.legend()
    plt.show()