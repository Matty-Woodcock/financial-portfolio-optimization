import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from test_metrics_calculator import MetricSummary, calculate_hhi_metric, calculate_mdd_metric, calculate_sharpe_ratio
from rl_approach.rl_reward import RewardFunction, NGSA2
from config.constants import ALL_SYMBOLS_WITH_CASH, RL_WINDOW_SIZE

def format_weights(tickers, weights):
    formatted_weights = {}
    for i in range(len(weights)):
        if weights[i] > 0:
            formatted_weights[tickers[i]] = float(weights[i])
    return formatted_weights

class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, prices: pd.DataFrame, reward_function: RewardFunction, volatilities, covariances = None, window_size = RL_WINDOW_SIZE, transaction_cost = 0.001):
        super(PortfolioEnv, self).__init__()

        self.reward_function = reward_function
        self.volatilities = volatilities
        self.covariances = covariances

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
        if self.reward_function.volatility_in_observation_space:
            observation_space_shape = (window_size + 1) * (self.n_assets - 1) + self.n_assets
        else:
            observation_space_shape = (window_size) * (self.n_assets - 1) + self.n_assets
        if self.reward_function.dcc_in_observation_space:
            n_risky = self.n_assets - 1
            observation_space_shape += n_risky * (n_risky - 1) // 2  # upper triangle of R_t (no diagonal)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_space_shape,), dtype=np.float32
        )

        self.weights, self.portfolio_value = None, None
    
        self.test_counter = 0

    def _softmax(self, x, temp=0.05):
        z = x / temp
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum()

    def _get_observation(self):
        window = self.prices[self.current_step - self.window_size : self.current_step]
        norm = window / (window[0] + 1e-8) - 1.0

        observation = np.concatenate([norm.flatten(), self.weights])
        #observation = norm.flatten()

        if self.reward_function.volatility_in_observation_space:
            daily_volatilities = self.volatilities[self.current_step - self.window_size]
            observation = np.concatenate([observation, daily_volatilities])

        if self.reward_function.dcc_in_observation_space and self.covariances is not None:
            cov = self.covariances[self.current_step - self.window_size]
            if not np.any(np.isnan(cov)):
                std = np.sqrt(np.maximum(np.diag(cov), 1e-10))
                corr = cov / np.outer(std, std)
                np.fill_diagonal(corr, 1.0)
                n = corr.shape[0]
                observation = np.concatenate([observation, corr[np.triu_indices(n, k=1)]])
            else:
                n_risky = self.n_assets - 1
                observation = np.concatenate([observation, np.zeros(n_risky * (n_risky - 1) // 2)])

        return observation.astype(np.float32)    
    
    def step(self, action):
        new_weights = self._softmax(action)
        # total_weight = np.sum(weights) + 1e-8
        # weights = weights / total_weight
        # self.weights = weights

        portfolio_value_before_step = self.portfolio_value

        if (self.observation and self.current_step > self.window_size) or (not self.observation and self.current_step > 0) or (not self.initial_env):
            turnover = np.sum(np.abs(new_weights - self.weights)) / 2.0
            self.cost = self.portfolio_value * turnover * self.transaction_cost
        else:
            turnover, self.cost = 0.0, 0.0 #This occurs on very first step when initially allocating portfolio
        
        #TO DO: Rewrite this functionality better
        portfolio_value_after_cost = self.portfolio_value - self.cost
        #transaction_cost_percentage = (self.cost / self.portfolio_value) * 100
        transaction_cost_percentage = self.cost * 100

        #Below can potentially be changed as we already calculated portfolio returns? But probably not.
        if self.current_step == 0:
            if self.prev_prices is None:
                raise ValueError("In PortfolioEnv, current_step starting from zero but prev_prices is None.")
            prev_prices = self.prev_prices
        else:
            prev_prices = self.prices[self.current_step - 1]
        current_prices = self.prices[self.current_step]

        self.test_counter += 1

        #Compute returns per asset and then combine with cash
        #asset_returns = (current_prices - prev_prices) / (prev_prices + 1e-8)
        asset_returns = (current_prices - prev_prices) / prev_prices
        asset_returns_with_cash = np.append(0, asset_returns)
        # print(asset_returns)
        # print(prev_prices)
        # print(current_prices)

        portfolio_return = np.dot(new_weights, asset_returns_with_cash)
        self.portfolio_value = portfolio_value_after_cost * (1 + portfolio_return)

        if not isinstance(self.reward_function, NGSA2):
            reward = self.reward_function(
                portfolio_value_before=portfolio_value_before_step,
                portfolio_value_after=self.portfolio_value,
                weights=new_weights,
                daily_volatilities = self.volatilities[self.current_step - self.window_size],
                daily_covariance = self.covariances[self.current_step - self.window_size] if self.covariances is not None else None,
                turnover=turnover,
                cost=self.cost,
            )
        else:
            reward = None

        portfolio_return = (self.portfolio_value - portfolio_value_before_step) / portfolio_value_before_step

        self.current_step += 1
        done = self.current_step >= len(self.prices)
        
        info = {"portfolio_value": self.portfolio_value, "weights": new_weights}

        self.weights = new_weights

        self.weights_history.append(self.weights)
        self.portfolio_value_history.append(self.portfolio_value)
        self.portfolio_return_history.append(portfolio_return)
        self.transaction_cost_percentage_history.append(transaction_cost_percentage)

        if not done and self.observation:
            obs = self._get_observation()
        else:
            obs = None

        return obs, reward, done, False, info
    
    def reset(self, seed=None, options=None, observation=True, initial_portfolio_value=1.0, initial_env=True, prev_prices=None): #This is called everytime
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_value = initial_portfolio_value
        self.weights = np.zeros(self.n_assets)
        self.weights[0] = 1.0

        self.observation = observation
        self.initial_env = initial_env
        self.prev_prices = prev_prices
        
        self.weights_history = []
        self.portfolio_value_history = []
        self.portfolio_return_history = []
        self.transaction_cost_percentage_history = []

        if self.observation:
            self.current_step = self.window_size
            obs = self._get_observation()
        else:
            self.current_step = 0
            obs = None

        self.test_counter = 0
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
            "portfolio_returns": MetricSummary(
                name="Portfolio Returns",
                reward_function_name=self.reward_function.__class__.__name__,
                direction=None,
                series=self.portfolio_return_history,
                aggregate=None
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
                aggregate= calculate_sharpe_ratio(
                    self.portfolio_return_history,
                    dates = self.dates[self.window_size:]
                )
            ),
            "garch_volatility": MetricSummary(
                name = self.reward_function.volatility_in_observation_space,
                reward_function_name = self.reward_function.__class__.__name__,
                direction = None,
                series = None,
                aggregate = None
            ),
            "transaction_cost_percentage": MetricSummary(
                name = "Total Transaction Cost Percentage",
                reward_function_name = self.reward_function.__class__.__name__,
                direction = None,
                series = self.transaction_cost_percentage_history,
                aggregate = np.sum(self.transaction_cost_percentage_history)
            ),
        }