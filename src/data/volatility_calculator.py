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


def dcc_volatility_calculator(prices_df, estimate_params=True):
    from data.dcc_garch_main import DCCGARCHModel

    burn_in_split_price_index = GARCH_BURN_IN_DAYS + 1
    train_split_price_index = int(len(prices_df[burn_in_split_price_index:]) * TRAIN_SPLIT) + burn_in_split_price_index
    eval_split_price_index = int(len(prices_df[burn_in_split_price_index:]) * EVAL_SPLIT) + burn_in_split_price_index

    returns_df = prices_df.pct_change().dropna()
    n_assets   = returns_df.shape[1]

    dcc             = DCCGARCHModel(n_assets, estimate_params=estimate_params)
    dcc_initialized = False
    all_volatilities = []
    all_covariances  = []

    for step, (_, row) in enumerate(returns_df.iterrows()):
        dcc.add_returns(row.values)

        if step == GARCH_BURN_IN_DAYS - 1 and not dcc_initialized:
            print(f"[DCC] Initialising DCC from burn-in history...")
            dcc.initialize_from_history()
            dcc_initialized = True

        vols, cov = dcc.forecast_covariance()
        all_volatilities.append(vols if vols is not None else np.full(n_assets, np.nan))
        all_covariances.append(cov  if cov  is not None else np.full((n_assets, n_assets), np.nan))

    all_volatilities = np.array(all_volatilities)
    all_covariances  = np.array(all_covariances)

    train_split_return_index = train_split_price_index - 1
    eval_split_return_index  = eval_split_price_index  - 1

    sl_train = slice(GARCH_BURN_IN_DAYS + RL_WINDOW_SIZE, train_split_return_index)
    sl_eval  = slice(train_split_return_index + RL_WINDOW_SIZE, eval_split_return_index)
    sl_test  = slice(eval_split_return_index  + RL_WINDOW_SIZE, None)

    return (
        all_volatilities[sl_train], all_volatilities[sl_eval], all_volatilities[sl_test],
        all_covariances[sl_train],  all_covariances[sl_eval],  all_covariances[sl_test],
    )