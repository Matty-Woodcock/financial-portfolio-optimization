from rl_env import PortfolioEnv
from rl_reward import RewardFunction, PureReturn
from price_data_reader_writer import load_or_update_price_data
from config.constants import GARCH_BURN_IN_DAYS

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

def train_and_test_one_policy(reward_function: RewardFunction, render=False):
    try:
        prices_df = load_or_update_price_data()
    except ValueError as error:
        print(f"Fetching Price Data Failed: {error}")
        return

    #Indexing below confirmed to be correct
    burn_in_split_index = GARCH_BURN_IN_DAYS
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

    garch_window_size = GARCH_BURN_IN_DAYS + 40 - 1 #Subtract one from RL window size to account for returns rather than price
    returns_df = prices_df.pct_change().dropna() #TO DO: Change later?

    pre_train_returns = returns_df.iloc[:garch_window_size].values
    # print(pre_train_returns.head())
    # print(pre_train_returns.tail())
    pre_eval_returns = returns_df.iloc[train_split_index - garch_window_size:train_split_index].values
    pre_test_returns = returns_df.iloc[eval_split_index - garch_window_size:eval_split_index].values
    assert len(pre_train_returns) == len(pre_eval_returns) == len(pre_test_returns) == 290

    train_env = PortfolioEnv(train_df, reward_function, pre_train_returns)
    eval_env = PortfolioEnv(eval_df, reward_function, pre_eval_returns)
    test_env = PortfolioEnv(test_df, reward_function, pre_test_returns)

    callback = EvalCallback(eval_env, 
                            best_model_save_path='./logs/',
                            log_path='./logs/', 
                            eval_freq=2000,
                            deterministic=True, 
                            render=False)


    model = PPO("MlpPolicy", train_env, verbose=1)

    print("Training Starting")
    model.learn(total_timesteps=8000, callback=callback)
    print("Training Complete")
    
    print(f"Training GARCH Average Runtimes: {train_env.garch_average_runtimes}")
    print(f"Evaluation GARCH Average Runtimes: {eval_env.garch_average_runtimes}")

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