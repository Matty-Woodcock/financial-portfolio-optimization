import numpy as np
from pymoo.core.problem import ElementwiseProblem

class PortfolioNSGA2Problem(ElementwiseProblem):

    def __init__(self, env, initial_env, initial_weights):
        super().__init__(
            n_var=env.n_assets,
            n_obj=2,
            xl=-10.0,
            xu=10.0
        )
        self.env = env
        self.initial_env = initial_env
        self.initial_weights = initial_weights

    def _evaluate(self, action, out, *args, **kwargs):
        obs, _ = self.env.reset(observation = False, initial_env = self.initial_env)
        if not self.initial_env:
            self.env.weights = self.initial_weights
        done = False

        while not done:
            obs, reward, done, _, _ = self.env.step(action)

        metrics = self.env.get_metric_summaries()

        objective_one = metrics["portfolio_value"].aggregate
        objective_two = metrics["mdd"].aggregate
        #objective_two = metrics["transaction_cost_percentage"].aggregate
        
        #pymoo tries to minimised by default
        out["F"] = [-objective_one, -objective_two]
        #out["F"] = [-objective_one, objective_two]

