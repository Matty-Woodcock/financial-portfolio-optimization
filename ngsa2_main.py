import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from price_data_reader_writer import load_or_update_price_data
from config.constants import GARCH_BURN_IN_DAYS
from rl_env import PortfolioEnv
from rl_reward import NGSA2
from ngsa2_problem import PortfolioNSGA2RollingProblem
from ngas2_utility import plot_pareto_front

def run_nsga(env, initial_env, initial_weights, prev_prices, pop_size=40, n_gen=20):

    problem = PortfolioNSGA2RollingProblem(env, initial_env, initial_weights, prev_prices)

    algorithm = NSGA2(pop_size=pop_size)

    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        verbose=True
    )

    return res

def train_and_test_one_ngsa2(render=False):
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

    #Configuration

    window_size = 40
    step_size = 10

    n_steps = len(test_df)
    assert n_steps > window_size

    weights_history = []
    portfolio_value_history = []
    portfolio_return_history = []
    transaction_cost_percentage_history = []

    #Training

    initial_env = True
    current_portfolio_value = 1.0
    train_prev_prices = eval_df.iloc[-1].values

    for start_index in range(0, n_steps - window_size, step_size):
        train_end = start_index + window_size
        test_end = min(train_end + step_size, n_steps)

        train_window = test_df.iloc[start_index:train_end]
        test_window = test_df.iloc[train_end:test_end]

        train_env = PortfolioEnv(train_window, NGSA2(), test_volatilities[start_index:train_end])

        if not initial_env:
            initial_weights = weights_history[-1]
        else:
            initial_weights = np.zeros(train_env.n_assets)
            initial_weights[0] = 1.0

        res = run_nsga(train_env, initial_env, initial_weights, train_prev_prices) #This is a result object
        pareto_actions = res.X
        pareto_objectives = res.F
    
        objective_one = pareto_objectives[:, 0]
        objective_two = pareto_objectives[:, 1]
        sign_correct_objective_one = -objective_one
        sign_correct_objective_two = -objective_two
        #sign_correct_objective_two = objective_two

        #Choose highest Sharpe solution
        # best_index = np.argmin(objective_one)
        # sorted_indices = np.argsort(objective_one)
        # one_quarter_index = sorted_indices[int(0.25 * len(sorted_indices))]
        # best_action = pareto_actions[best_index]
        
        #Normalise objectives to [0,1] so neither dominates by scale
        obj_min = pareto_objectives.min(axis=0)
        obj_max = pareto_objectives.max(axis=0)
        obj_range = obj_max - obj_min + 1e-8  #Avoids division by zero
        normalised = (pareto_objectives - obj_min) / obj_range

        #Utopia point is (0, 0) after normalisation
        #distances = np.linalg.norm(normalised, axis=1)
        distances = np.linalg.norm(normalised * np.array([0.65, 0.35]), axis=1)
        best_index = np.argmin(distances)
        best_action = pareto_actions[best_index]

        # if start_index == 0:
        #     plot_pareto_front(sign_correct_objective_one, sign_correct_objective_two, best_index)

        #Testing

        test_prev_prices = train_window.iloc[-1].values 

        test_env = PortfolioEnv(test_window, NGSA2(), test_volatilities[train_end:test_end])

        obs, _ = test_env.reset(observation=False, initial_portfolio_value=current_portfolio_value, initial_env=initial_env, prev_prices=test_prev_prices)
        if not initial_env:
            test_env.weights = weights_history[-1]
        initial_env = False
        done = False

        while not done:
            _, _, done, _, _ = test_env.step(best_action)

        test_metric_summaries = test_env.get_metric_summaries()
        test_length = len(test_window)
        
        weights_history.extend(test_metric_summaries["portfolio_weights"].series[-test_length:])  
        portfolio_value_history.extend(test_metric_summaries["portfolio_value"].series[-test_length:])  
        portfolio_return_history.extend(test_env.portfolio_return_history[-test_length:])  
        transaction_cost_percentage_history.extend(test_metric_summaries["transaction_cost_percentage"].series[-test_length:])

        current_portfolio_value = portfolio_value_history[-1]
        train_prev_prices = train_window.iloc[step_size - 1].values

    test_env.weights_history = weights_history
    test_env.portfolio_value_history = portfolio_value_history
    test_env.portfolio_return_history = portfolio_return_history
    test_env.transaction_cost_percentage_history = transaction_cost_percentage_history
    test_env.prices = test_df.values

    return test_env.get_metric_summaries()

if __name__ == "__main__":
    train_and_test_one_ngsa2(render=False)