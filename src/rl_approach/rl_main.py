import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_approach.rl_env import PortfolioEnv
from rl_approach.rl_reward import RewardFunction, PureReturn
from data.price_data_reader_writer import load_or_update_price_data
from config.constants import GARCH_BURN_IN_DAYS,  TRAIN_SPLIT, EVAL_SPLIT, RL_WINDOW_SIZE

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

def train_and_test_one_policy(reward_function: RewardFunction, render=False, seed=None):
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

    train_env = PortfolioEnv(train_df, reward_function, train_volatilities)
    eval_env = PortfolioEnv(eval_df, reward_function, eval_volatilities)
    test_env = PortfolioEnv(test_df, reward_function, test_volatilities)

    callback = EvalCallback(eval_env, 
                            best_model_save_path='./logs/',
                            log_path='./logs/', 
                            eval_freq=2000,
                            deterministic=True, 
                            render=False)

    model = PPO("MlpPolicy", train_env, verbose=1, ent_coef=0.05, seed=seed, device="cpu") #CPU is faster than GPU for this task

    print("Training Starting")
    model.learn(total_timesteps=20000, callback=callback)
    print("Training Complete")

    print("\nLoading Best Model for Testing")
    best_model = PPO.load("./logs/best_model") #Automatically appends .zip

    print("\nTesting Best Model")
    obs, info = test_env.reset()
    done = False

    while not done:
        action, _states = best_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        if render:
            test_env.render()

    assert test_env.test_counter == len(test_df) - RL_WINDOW_SIZE

    return test_env.get_metric_summaries()

if __name__ == "__main__":
    train_and_test_one_policy(PureReturn(), render=True)

#First trading day of test period
# [ 45.03986489  99.90579164  62.17089817  90.8635     250.31431039
#  168.9674853   96.64447269 115.83        36.65840437 347.12888221
#   57.94640514]
# [ 44.7875209   97.76256747  61.03703564  88.961      249.00679125
#  166.5797565   96.31277811 115.65        36.67427038 337.8984206
#   57.8190815 ]

#290 [0.02263598] Volatility