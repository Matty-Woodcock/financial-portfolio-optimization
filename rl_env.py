import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from test_metrics_calculator import calculate_hhi_metric

def format_weights(tickers, weights):
    formatted_weights = {}
    for i in range(len(weights)):
        if weights[i] > 0:
            formatted_weights[tickers[i]] = float(weights[i])
    return formatted_weights

class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, prices: pd.DataFrame, window_size=40, transaction_cost = 0.001):
        super(PortfolioEnv, self).__init__()

        self.prices = prices.values  # shape (T, 2)
        self.dates = prices.index
        self.tickers = ["CASH"] + list(prices.columns)
        self.window_size = window_size
        self.current_step = window_size
        self.transaction_cost = transaction_cost

        #TO DO: Should I treat cash as static?
        self.n_assets = self.prices.shape[1] + 1

        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.n_assets,), 
            dtype=np.float32
        )
        # self.action_space = spaces.Box(
        #     low=0.0,
        #     high=1.0,
        #     shape=(self.n_assets,), 
        #     dtype=np.float32
        # )

        # Observation: normalized prices for both assets over past N weeks
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size * (self.n_assets - 1),), dtype=np.float32
        )

        self.weights, self.portfolio_value = None, None
        self.weights_history = []

    def _softmax(self, x, temp=0.05):
        z = x / temp
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum()

    def _get_observation(self):
        window = self.prices[self.current_step - self.window_size : self.current_step]
        norm = window / (window[0] + 1e-8) - 1.0
        return norm.flatten().astype(np.float32)    
    
    def step(self, action):
        #TO DO: Is sparsemax worth pursuing?
        new_weights = self._softmax(action)
        # total_weight = np.sum(weights) + 1e-8
        # weights = weights / total_weight
        # self.weights = weights

        portfolio_value_before_step = self.portfolio_value

        if self.current_step > self.window_size:
            turnover = np.sum(np.abs(new_weights - self.weights)) / 2.0
            self.cost = self.portfolio_value * turnover * self.transaction_cost
        else:
            self.cost = 0.0 #This occurs on very first step when initially allocating portfolio
        
        #TO DO: Rewrite this functionality better
        portfolio_value_after_cost = self.portfolio_value - self.cost

        #Below can potentially be changed as we already calculated portfolio returns? But probably not.
        prev_prices = self.prices[self.current_step - 1]
        current_prices = self.prices[self.current_step]

        #Compute returns per asset and then combine with cash
        asset_returns = (current_prices - prev_prices) / (prev_prices + 1e-8)
        asset_returns_with_cash = np.append(0, asset_returns)

        portfolio_return = np.dot(new_weights, asset_returns_with_cash)
        self.portfolio_value = portfolio_value_after_cost * (1 + portfolio_return)

        reward = (self.portfolio_value - portfolio_value_before_step) / portfolio_value_before_step
        
        self.current_step += 1
        done = self.current_step >= len(self.prices)

        obs = self._get_observation()
        info = {"portfolio_value": self.portfolio_value, "weights": new_weights}

        self.weights = new_weights
        self.weights_history.append(self.weights)

        return obs, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_value = 1.0
        self.weights = np.zeros(self.n_assets)
        self.weights[0] = 1.0

        obs = self._get_observation()
        info = {}
        return obs, info

    def render(self):
        current_date_str = self.dates[self.current_step - 1].strftime('%Y-%m-%d')
        print(f"{current_date_str} (Day {self.current_step}): Portfolio Value: {self.portfolio_value:.4f}, Cost: {self.cost}, {format_weights(self.tickers, self.weights)}")    

    def get_metrics(self) -> list[int]:
        return [self.portfolio_value, calculate_hhi_metric(self.weights_history)]