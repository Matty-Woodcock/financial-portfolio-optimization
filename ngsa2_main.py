import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from price_data_reader_writer import load_or_update_price_data
from config.constants import GARCH_BURN_IN_DAYS
from rl_env import PortfolioEnv
from rl_reward import RewardFunction, PureReturn
from ngsa2_problem import PortfolioNSGA2Problem

def run_nsga(env, pop_size=60, n_gen=20):

    problem = PortfolioNSGA2Problem(env)

    algorithm = NSGA2(pop_size=pop_size)

    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        verbose=True
    )

    return res

def plot_pareto_front(objective_one, objective_two):
    plt.figure(figsize=(8,6))
    plt.scatter(objective_one, objective_two, c='blue', label='Pareto Solutions')
    plt.xlabel("Objective One")
    plt.ylabel("Objective Two")
    plt.title("Pareto Front")
    plt.grid(True)

    plt.legend()
    plt.show()

def train_and_test_one_ngsa2(reward_function: RewardFunction, render=False):
    try:
        prices_df, train_volatilities, eval_volatilities, test_volatilities = load_or_update_price_data()
    except ValueError as error:
        print(f"Fetching Price Data Failed: {error}")
        return

    #Indexing below confirmed to be correct
    burn_in_split_index = GARCH_BURN_IN_DAYS + 1 #Prices Length = Returns Length + 1
    train_split_index = int(len(prices_df[burn_in_split_index:]) * 0.7) + burn_in_split_index
    eval_split_index = int(len(prices_df[burn_in_split_index:]) * 0.85) + burn_in_split_index
    
    burn_in_df = prices_df[:burn_in_split_index]
    train_df = prices_df.iloc[burn_in_split_index:train_split_index]
    eval_df = prices_df.iloc[train_split_index:eval_split_index]
    test_df = prices_df.iloc[eval_split_index:]

    print(f"Burning in on {len(burn_in_df)} days.") #251
    print(f"Training on {len(train_df)} days.") #1148
    print(f"Evaluating on {len(eval_df)} days.") #246
    print(f"Testing on {len(test_df)} days.") #247

    window_size = 40
    n_steps = len(test_df)

    assert n_steps > window_size

    weights_history = []
    portfolio_value_history = []
    portfolio_return_history = []
    transaction_cost_percentage_history = []

    #Training

    initial_env = True
    current_portfolio_value = 1.0
    step_size = 5

    for start_index in range(0, n_steps - window_size, step_size):
        train_end = start_index + window_size #TO DO: Correct? Not +1?
        test_end = min(train_end + step_size, n_steps)

        train_window = test_df.iloc[start_index:train_end]
        test_window = test_df.iloc[train_end:test_end]

        train_env = PortfolioEnv(train_window, PureReturn(), test_volatilities[start_index:train_end])

        res = run_nsga(train_env) #This is a result object
        pareto_actions = res.X
        pareto_objectives = res.F
    
        objective_one = pareto_objectives[:, 0]
        objective_two = pareto_objectives[:, 1]
        sign_correct_objective_one = -objective_one
        sign_correct_objective_two = -objective_two

        #Choose highest Sharpe solution
        best_index = np.argmin(objective_one)
        best_action = pareto_actions[best_index]

        # print(pareto_objectives[best_index]) #TO DO: All in cash or error due to no change?

        # plot_pareto_front(sign_correct_objective_one, sign_correct_objective_two)

        #Testing

        test_env = PortfolioEnv(test_window, PureReturn(), test_volatilities[train_end:test_end])

        obs, _ = test_env.reset(observation=False, initial_portfolio_value=current_portfolio_value, initial_env=initial_env)
        if not initial_env:
            test_env.weights = weights_history[-1]
        initial_env = False
        done = False

        while not done:
            obs, _, done, _, info = test_env.step(best_action)

        test_metric_summaries = test_env.get_metric_summaries()
        test_length = len(test_window)
        
        weights_history.extend(test_metric_summaries["portfolio_weights"].series[-test_length:])  
        portfolio_value_history.extend(test_metric_summaries["portfolio_value"].series[-test_length:])  
        portfolio_return_history.extend(test_env.portfolio_return_history[-test_length:])  
        transaction_cost_percentage_history.extend(test_metric_summaries["transaction_cost_percentage"].series[-test_length:])

        current_portfolio_value = portfolio_value_history[-1]

    test_env.weights_history = weights_history
    test_env.portfolio_value_history = portfolio_value_history
    test_env.portfolio_return_history = portfolio_return_history
    test_env.transaction_cost_percentage_history = transaction_cost_percentage_history
    test_env.prices = test_df.values

    return test_env.get_metric_summaries()

if __name__ == "__main__":
    train_and_test_one_ngsa2(PureReturn(), render=False)

#TO DO: Add transaction cost to training the same way I have done with testing