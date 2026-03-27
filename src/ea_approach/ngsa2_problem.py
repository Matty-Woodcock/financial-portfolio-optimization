import numpy as np
from pymoo.core.problem import ElementwiseProblem

class PortfolioNSGA2RollingProblem(ElementwiseProblem):

    def __init__(self, env, initial_env, initial_weights, prev_prices):
        super().__init__(
            n_var=env.n_assets,
            n_obj=2,
            xl=-10.0,
            xu=10.0
        )
        self.env = env
        self.initial_env = initial_env
        self.initial_weights = initial_weights
        self.prev_prices = prev_prices

    def _evaluate(self, action, out, *args, **kwargs):
        obs, _ = self.env.reset(observation = False, initial_env = self.initial_env, prev_prices=self.prev_prices)
        if not self.initial_env:
            self.env.weights = self.initial_weights.copy()
        done = False

        while not done:
            obs, reward, done, _, _ = self.env.step(action)

        metrics = self.env.get_metric_summaries()

        objective_one = metrics["portfolio_value"].aggregate
        objective_two = metrics["mdd"].aggregate
        
        #pymoo tries to minimised by default
        out["F"] = [-objective_one, -objective_two]

class PortfolioNSGA2StaticProblem(ElementwiseProblem):

    def __init__(self, env):
        super().__init__(
            n_var=env.n_assets,
            n_obj=2,
            xl=-10.0,
            xu=10.0
        )
        self.env = env

    def _evaluate(self, action, out, *args, **kwargs):
        obs, _ = self.env.reset()

        done = False

        while not done:
            obs, reward, done, _, _ = self.env.step(action)

        metrics = self.env.get_metric_summaries()

        objective_one = metrics["portfolio_value"].aggregate
        objective_two = metrics["mdd"].aggregate
        
        #pymoo tries to minimised by default
        out["F"] = [-objective_one, -objective_two]