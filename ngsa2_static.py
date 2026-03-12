import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from price_data_reader_writer import load_or_update_price_data
from config.constants import GARCH_BURN_IN_DAYS, TRAIN_SPLIT, EVAL_SPLIT, RL_WINDOW_SIZE
from rl_env import PortfolioEnv
from rl_reward import NGSA2
from ngsa2_problem import PortfolioNSGA2StaticProblem
from ngas2_utility import plot_pareto_front

def run_nsga(env, pop_size=80, n_gen=50):

    problem = PortfolioNSGA2StaticProblem(env)

    algorithm = NSGA2(pop_size=pop_size)

    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        verbose=True
    )

    return res

def train_and_test_one_static_ngsa2(render=False):
    try:
        prices_df, train_volatilities, eval_volatilities, test_volatilities = load_or_update_price_data()
    except ValueError as error:
        print(f"Fetching Price Data Failed: {error}")
        return

    #Indexing below confirmed to be correct
    burn_in_split_index = GARCH_BURN_IN_DAYS + 1 #Prices Length = Returns Length + 1
    train_split_index = int(len(prices_df[burn_in_split_index:]) * TRAIN_SPLIT) + burn_in_split_index
    eval_split_index = int(len(prices_df[burn_in_split_index:]) * EVAL_SPLIT) + burn_in_split_index
    
    burn_in_df = prices_df[:burn_in_split_index]
    train_df = prices_df.iloc[burn_in_split_index:train_split_index]
    eval_df = prices_df.iloc[train_split_index:eval_split_index]
    test_df = prices_df.iloc[eval_split_index:]

    print(f"Burning in on {len(burn_in_df)} days.") #251
    print(f"Training on {len(train_df)} days.") #1148
    print(f"Evaluating on {len(eval_df)} days.") #246
    print(f"Testing on {len(test_df)} days.") #247

    train_env = PortfolioEnv(train_df, NGSA2(), train_volatilities)
    eval_env = PortfolioEnv(eval_df, NGSA2(), eval_volatilities)
    test_env = PortfolioEnv(test_df, NGSA2(), test_volatilities)
    
    #Training

    res = run_nsga(eval_env) #This is a result object

    pareto_actions = res.X
    pareto_objectives = res.F
    
    objective_one = pareto_objectives[:, 0]
    objective_two = pareto_objectives[:, 1]
    sign_correct_objective_one = -objective_one
    sign_correct_objective_two = -objective_two

    #Choose highest Sharpe solution
    best_index = np.argmin(objective_one)
    best_action = pareto_actions[best_index]

    print(pareto_objectives[best_index])

    plot_pareto_front(sign_correct_objective_one, sign_correct_objective_two, best_index)

    #Testing

    obs, _ = test_env.reset()
    done = False

    while not done:
        _, _, done, _, _ = test_env.step(best_action)

    ngsa2_metric_summaries = test_env.get_metric_summaries()
    assert len(ngsa2_metric_summaries["portfolio_value"].series) == len(test_df) - RL_WINDOW_SIZE
    return ngsa2_metric_summaries

if __name__ == "__main__":
    train_and_test_one_static_ngsa2(render=False)