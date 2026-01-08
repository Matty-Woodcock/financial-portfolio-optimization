import numpy as np  
import matplotlib.pyplot as plt

#Herfindahl–Hirschman Index (HHI)
def calculate_hhi_metric(weights_history):
    hhi_history = []
    for daily_weights in weights_history: 
        hhi_history.append(np.sum(np.square(daily_weights)))
    plot_metric(hhi_history)
    if np.mean(hhi_history) > 0.2: #TO DO: Remove this testing
        print(weights_history)
    return np.mean(hhi_history)

def plot_metric(y):
    y = np.asarray(y)
    x = np.arange(len(y))

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Day")
    plt.ylabel("Value")
    plt.title("Metric Over Time")
    plt.show()
