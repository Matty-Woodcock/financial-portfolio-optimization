import numpy as np

from config.constants import GARCH_BURN_IN_DAYS, TRAIN_SPLIT, EVAL_SPLIT, RL_WINDOW_SIZE
from data.garch_main import MultiAssetGARCH

def volatility_calculator(prices_df):
    burn_in_split_price_index = GARCH_BURN_IN_DAYS + 1 #Prices Length = Returns Length + 1
    train_split_price_index = int(len(prices_df[burn_in_split_price_index:]) * TRAIN_SPLIT) + burn_in_split_price_index
    eval_split_price_index = int(len(prices_df[burn_in_split_price_index:]) * EVAL_SPLIT) + burn_in_split_price_index
    
    burn_in_df = prices_df[:burn_in_split_price_index]
    train_df = prices_df.iloc[burn_in_split_price_index:train_split_price_index]
    eval_df = prices_df.iloc[train_split_price_index:eval_split_price_index]
    test_df = prices_df.iloc[eval_split_price_index:]

    print(f"Burning in on {len(burn_in_df)} days.") #251
    print(f"Training on {len(train_df)} days.") #1148
    print(f"Evaluating on {len(eval_df)} days.") #246
    print(f"Testing on {len(test_df)} days.") #247

    returns_df = prices_df.pct_change().dropna()

    multi_asset_garch = MultiAssetGARCH(n_assets = returns_df.shape[1])
    volatilities = []

    for _, row in returns_df.iterrows():
        multi_asset_garch.add_returns(row.values)
        daily_volatilities = multi_asset_garch.forecast_volatilities()
        volatilities.append(daily_volatilities)

    volatilities = np.array(volatilities)

    train_split_return_index = train_split_price_index - 1 #Returns Length = Prices Length - 1
    eval_split_return_index = eval_split_price_index - 1

    train_volatilities = volatilities[GARCH_BURN_IN_DAYS + RL_WINDOW_SIZE:train_split_return_index]
    eval_volatilities  = volatilities[train_split_return_index + RL_WINDOW_SIZE:eval_split_return_index]
    test_volatilities  = volatilities[eval_split_return_index + RL_WINDOW_SIZE:]

    #TO DO: How to test this? Add assert with config variables

    return train_volatilities, eval_volatilities, test_volatilities