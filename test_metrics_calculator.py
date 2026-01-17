import numpy as np  
import matplotlib.pyplot as plt

from dataclasses import dataclass

#TO DO: Split this file up?

@dataclass
class MetricSummary:
    name: str
    direction: int
    series: list[float]
    aggregate: float

#Herfindahl–Hirschman Index (HHI)
def calculate_hhi_metric(weights_history): #TO DO: Implement Vectorization
    hhi_history = []
    for daily_weights in weights_history: 
        hhi_history.append(np.sum(np.square(daily_weights)))
    # plot_metric(hhi_history)
    # if np.mean(hhi_history) > 0.1: #TO DO: Remove this testing
    #     print(weights_history)
    return hhi_history, np.mean(hhi_history)

#Maximum Drawdown (MDD)
def calculate_mdd_metric(portfolio_value_history): #TO DO: Implement Vectorization
    mdd_history = []
    peak_porfolio_value = 1
    for daily_portfolio_value in portfolio_value_history:
        peak_porfolio_value = max(peak_porfolio_value, daily_portfolio_value)
        mdd_history.append((daily_portfolio_value - peak_porfolio_value) / peak_porfolio_value)
    #plot_metric(mdd_history) #TO DO: Remove this testing
    return mdd_history, min(mdd_history)

# def plot_metric(y):
#     y = np.asarray(y)
#     x = np.arange(len(y))

#     plt.figure()
#     plt.plot(x, y)
#     plt.xlabel("Day")
#     plt.ylabel("Value")
#     plt.title("Metric Over Time")
#     plt.show()



