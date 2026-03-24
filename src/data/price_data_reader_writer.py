import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data.price_data_fetcher import fetch_tiingo_price_data_for_all_symbols
from data.volatility_calculator import volatility_calculator, dcc_volatility_calculator

CSV_FILE = os.path.join(BASE_DIR, "data", "price_data.csv")
COV_DIR  = os.path.join(BASE_DIR, "data", "covariances")

def load_or_update_price_data(force_refresh=False):
    """
    Loads price data from CSV if it exists, otherwise fetches from Tiingo and saves to CSV.
    """
    if os.path.exists(CSV_FILE) and not force_refresh:
        print(f"Loading price data from CSV...")
        prices_df = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
        train_volatilities = np.load(os.path.join(BASE_DIR, "data", "volatilities", "train_volatilities.npy"), allow_pickle=True)
        eval_volatilities  = np.load(os.path.join(BASE_DIR, "data", "volatilities", "eval_volatilities.npy"),  allow_pickle=True)
        test_volatilities  = np.load(os.path.join(BASE_DIR, "data", "volatilities", "test_volatilities.npy"),  allow_pickle=True)
    else:
        print("Fetching updated price data from Tiingo API...")
        try:
            prices_df = fetch_tiingo_price_data_for_all_symbols()
        except ValueError as error:
            raise error
        if prices_df is not None:
            prices_df.to_csv(CSV_FILE)
            print(f"Price Data Saved To CSV")
        train_volatilities, eval_volatilities, test_volatilities = volatility_calculator(prices_df)
        np.save(os.path.join(BASE_DIR, "data", "volatilities", "train_volatilities.npy"), train_volatilities)
        np.save(os.path.join(BASE_DIR, "data", "volatilities", "eval_volatilities.npy"),  eval_volatilities)
        np.save(os.path.join(BASE_DIR, "data", "volatilities", "test_volatilities.npy"),  test_volatilities)

    return prices_df, train_volatilities, eval_volatilities, test_volatilities

def load_or_update_dcc_data(force_refresh=False, estimate_params=True):
    train_cov_path = os.path.join(COV_DIR, "train_covariances.npy")

    if os.path.exists(train_cov_path) and not force_refresh:
        print("Loading DCC covariance data from cache...")
        prices_df          = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
        train_volatilities = np.load(os.path.join(COV_DIR, "train_volatilities.npy"))
        eval_volatilities  = np.load(os.path.join(COV_DIR, "eval_volatilities.npy"))
        test_volatilities  = np.load(os.path.join(COV_DIR, "test_volatilities.npy"))
        train_covariances  = np.load(os.path.join(COV_DIR, "train_covariances.npy"))
        eval_covariances   = np.load(os.path.join(COV_DIR, "eval_covariances.npy"))
        test_covariances   = np.load(os.path.join(COV_DIR, "test_covariances.npy"))
    else:
        print("Computing DCC-GARCH covariance matrices (this may take a few minutes)...")
        if not os.path.exists(CSV_FILE) or force_refresh:
            try:
                prices_df = fetch_tiingo_price_data_for_all_symbols()
            except ValueError as error:
                raise error
            prices_df.to_csv(CSV_FILE)
        else:
            prices_df = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)

        os.makedirs(COV_DIR, exist_ok=True)
        (
            train_volatilities, eval_volatilities,  test_volatilities,
            train_covariances,  eval_covariances,   test_covariances,
        ) = dcc_volatility_calculator(prices_df, estimate_params=estimate_params)
        np.save(os.path.join(COV_DIR, "train_volatilities.npy"), train_volatilities)
        np.save(os.path.join(COV_DIR, "eval_volatilities.npy"),  eval_volatilities)
        np.save(os.path.join(COV_DIR, "test_volatilities.npy"),  test_volatilities)
        np.save(os.path.join(COV_DIR, "train_covariances.npy"),  train_covariances)
        np.save(os.path.join(COV_DIR, "eval_covariances.npy"),   eval_covariances)
        np.save(os.path.join(COV_DIR, "test_covariances.npy"),   test_covariances)
        print("DCC covariance data saved to cache.")

    return (
        prices_df,
        train_volatilities, eval_volatilities, test_volatilities,
        train_covariances,  eval_covariances,  test_covariances,
    )


if __name__ == "__main__":
    prices = load_or_update_price_data(force_refresh=True)
    load_or_update_dcc_data(force_refresh=True)
    #print(prices.head())