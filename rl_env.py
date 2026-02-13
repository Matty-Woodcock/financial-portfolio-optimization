import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from test_metrics_calculator import MetricSummary, calculate_hhi_metric, calculate_mdd_metric, calculate_sharpe_ratio
from rl_reward import RewardFunction
from config.constants import ALL_SYMBOLS_WITH_CASH

def format_weights(tickers, weights):
    formatted_weights = {}
    for i in range(len(weights)):
        if weights[i] > 0:
            formatted_weights[tickers[i]] = float(weights[i])
    return formatted_weights

class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, prices: pd.DataFrame, reward_function: RewardFunction, volatilities, window_size=40, transaction_cost = 0.001):
        super(PortfolioEnv, self).__init__()

        self.reward_function = reward_function
        self.volatilities = volatilities

        self.prices = prices.values  # shape (T, 2)
        self.dates = prices.index
        self.tickers = ALL_SYMBOLS_WITH_CASH
        self.window_size = window_size
        self.current_step = window_size
        self.transaction_cost = transaction_cost

        self.n_assets = self.prices.shape[1] + 1

        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.n_assets,), 
            dtype=np.float32
        )

        # Observation: normalized prices for both assets over past N weeks
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=((window_size) * (self.n_assets - 1),), dtype=np.float32 #TO DO: Needs to be changed
        )

        self.weights, self.portfolio_value = None, None
        self.weights_history = []
        self.portfolio_value_history = []
        self.portfolio_return_history = []

        self.test_counter = 0

    def _softmax(self, x, temp=0.05):
        z = x / temp
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum()

    def _get_observation(self):
        window = self.prices[self.current_step - self.window_size : self.current_step]
        # print(self.current_step - self.window_size, self.current_step)
        # print(len(window))
        norm = window / (window[0] + 1e-8) - 1.0

        #daily_volatilities = self.volatilities[self.current_step - self.window_size]
        #daily_volatilities = np.array([1])
        #observation = np.concatenate([norm.flatten(), daily_volatilities])
        observation = norm.flatten()
        self.test_counter += 1
        if self.test_counter == 1:
            print(len(window))
            print(self.current_step)
            input("Z")

        if self.test_counter == 1109:
            print(len(self.prices))
            print(self.current_step)
            input("Y")

        # print(observation)
        # print(self.garch_model.total_steps, volatility)
        
        # if self.garch_model.total_steps == 290:
        #     print(self.x)
        #     print(self.garch_model.total_steps, volatility)
        #     print(len(self.prices))
        #     input("X")
        # else:
        #     self.x = self.garch_model.total_steps

        # print(observation)
        # print(volatilities)
        # input("X")

        return observation.astype(np.float32)    
    
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
            turnover, self.cost = 0.0, 0.0 #This occurs on very first step when initially allocating portfolio
        
        #TO DO: Rewrite this functionality better
        portfolio_value_after_cost = self.portfolio_value - self.cost

        #Below can potentially be changed as we already calculated portfolio returns? But probably not.
        prev_prices = self.prices[self.current_step - 1]
        current_prices = self.prices[self.current_step]

        #Compute returns per asset and then combine with cash
        #asset_returns = (current_prices - prev_prices) / (prev_prices + 1e-8)
        asset_returns = (current_prices - prev_prices) / prev_prices
        asset_returns_with_cash = np.append(0, asset_returns)
        # print(asset_returns)
        # print(prev_prices)
        # print(current_prices)

        portfolio_return = np.dot(new_weights, asset_returns_with_cash)
        self.portfolio_value = portfolio_value_after_cost * (1 + portfolio_return)

        reward = self.reward_function(
            portfolio_value_before=portfolio_value_before_step,
            portfolio_value_after=self.portfolio_value,
            weights=new_weights,
            prev_weights=self.weights,
            turnover=turnover,
            cost=self.cost,
        )

        portfolio_return = (self.portfolio_value - portfolio_value_before_step) / portfolio_value_before_step
        # assert portfolio_return == reward

        self.current_step += 1
        done = self.current_step >= len(self.prices)

        obs = self._get_observation()
        info = {"portfolio_value": self.portfolio_value, "weights": new_weights}

        self.weights = new_weights

        self.weights_history.append(self.weights)
        self.portfolio_value_history.append(self.portfolio_value)
        self.portfolio_return_history.append(portfolio_return)

        return obs, reward, done, False, info
    
    def reset(self, seed=None, options=None): #TO DO: Experiment with how often this is called
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_value = 1.0
        self.weights = np.zeros(self.n_assets)
        self.weights[0] = 1.0
        print(self.test_counter)
        input("X")
        self.test_counter = 0
        obs = self._get_observation()
        info = {}
        return obs, info

    def render(self):
        current_date_str = self.dates[self.current_step - 1].strftime('%Y-%m-%d')
        print(f"{current_date_str} (Day {self.current_step}): Portfolio Value: {self.portfolio_value:.4f}, Cost: {self.cost}, {format_weights(self.tickers, self.weights)} {self.current_step - 2}")

    def get_metric_summaries(self) -> dict[str, MetricSummary]:
        #return [self.portfolio_value, calculate_hhi_metric(self.weights_history), calculate_mdd_metric(self.portfolio_value_history)]

        hhi_series, mean_hhi = calculate_hhi_metric(self.weights_history)
        mdd_series, max_drawdown = calculate_mdd_metric(self.portfolio_value_history)

        return {
            "portfolio_value": MetricSummary(
                name = "Final Portfolio Value",
                reward_function_name = self.reward_function.__class__.__name__,
                direction = +1,
                series = self.portfolio_value_history,
                aggregate = self.portfolio_value_history[-1]
            ),
            "hhi": MetricSummary(
                name = "Mean HHI",
                reward_function_name = self.reward_function.__class__.__name__,
                direction = -1,
                series = hhi_series,
                aggregate=mean_hhi
            ),
            "mdd": MetricSummary(
                name = "Max Drawdown",
                reward_function_name = self.reward_function.__class__.__name__,
                direction = +1,
                series = mdd_series,
                aggregate = max_drawdown
            ),
            "portfolio_weights": MetricSummary(
                name = "Portfolio Weights",
                reward_function_name = self.reward_function.__class__.__name__,
                direction = None,
                series = self.weights_history,
                aggregate = None
            ),
            "ticker_prices": MetricSummary(
                name = "Ticker Prices",
                reward_function_name = self.reward_function.__class__.__name__,
                direction = None,
                series = self.prices,
                aggregate = None
            ),
            "sharpe_ratio": MetricSummary(
                name = "Sharpe Ratio",
                reward_function_name = self.reward_function.__class__.__name__,
                direction = None,
                series = None,
                aggregate = calculate_sharpe_ratio(self.portfolio_return_history)
            )
        }