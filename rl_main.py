from rl_env import PortfolioEnv
from price_data_fetcher import fetch_tiingo_price_data_for_all_symbols
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

if __name__ == "__main__":
    prices_df = fetch_tiingo_price_data_for_all_symbols()
    
    train_split_index = int(len(prices_df) * 0.7)
    eval_split_index = int(len(prices_df) * 0.85)
    
    train_df = prices_df.iloc[:train_split_index]
    eval_df = prices_df.iloc[train_split_index:eval_split_index]
    test_df = prices_df.iloc[eval_split_index:]
    
    print(f"Training on {len(train_df)} days.") #810
    print(f"Evaluating on {len(eval_df)} days.") #174
    print(f"Testing on {len(test_df)} days.") #174

    train_env = PortfolioEnv(train_df, window_size=40)
    eval_env = PortfolioEnv(eval_df, window_size=40)
    test_env = PortfolioEnv(test_df, window_size=40)

    callback = EvalCallback(eval_env, 
                            best_model_save_path='./logs/',
                            log_path='./logs/', 
                            eval_freq=1000,
                            deterministic=True, 
                            render=False)


    model = PPO("MlpPolicy", train_env, verbose=1)

    print("Training Starting")
    model.learn(total_timesteps=10000, callback=callback)
    print("Training Complete")


    print("\nLoading Best Model for Testing")
    best_model = PPO.load("./logs/best_model") #Automatically appends .zip

    print("\nTesting Best Model")
    obs, info = test_env.reset()
    done = False

    while not done:
        action, _states = best_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        test_env.render()