import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data.price_data_fetcher import fetch_tiingo_price_data_for_all_symbols
from data.volatility_calculator import volatility_calculator
from config.constants import DATA_END

CSV_FILE = os.path.join(BASE_DIR, "data", "price_data.csv")

def load_or_update_price_data(force_refresh=False):
    """
    Loads price data from CSV if it exists, otherwise fetches from Tiingo and saves to CSV.
    Returns prices_df (DATA_START to DATA_END) and extra_test_df (DATA_END onwards) separately.
    """
    if os.path.exists(CSV_FILE) and not force_refresh:
        print(f"Loading price data from CSV...")
        full_df = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
        train_volatilities = np.load(os.path.join(BASE_DIR, "data", "volatilities", "train_volatilities.npy"), allow_pickle=True)
        eval_volatilities = np.load(os.path.join(BASE_DIR, "data", "volatilities", "eval_volatilities.npy"),  allow_pickle=True)
        test_volatilities = np.load(os.path.join(BASE_DIR, "data", "volatilities", "test_volatilities.npy"),  allow_pickle=True)
        extra_test_volatilities = np.load(os.path.join(BASE_DIR, "data", "volatilities", "extra_test_volatilities.npy"), allow_pickle=True)
    else:
        print("Fetching updated price data from Tiingo API...")
        try:
            full_df = fetch_tiingo_price_data_for_all_symbols()
        except ValueError as error:
            raise error
        if full_df is not None:
            full_df.to_csv(CSV_FILE)
            print(f"Price Data Saved To CSV")
        train_volatilities, eval_volatilities, test_volatilities, extra_test_volatilities = volatility_calculator(full_df)
        np.save(os.path.join(BASE_DIR, "data", "volatilities", "train_volatilities.npy"), train_volatilities)
        np.save(os.path.join(BASE_DIR, "data", "volatilities", "eval_volatilities.npy"),  eval_volatilities)
        np.save(os.path.join(BASE_DIR, "data", "volatilities", "test_volatilities.npy"),  test_volatilities)
        np.save(os.path.join(BASE_DIR, "data", "volatilities", "extra_test_volatilities.npy"), extra_test_volatilities)

    data_end_index = full_df.index.get_indexer([DATA_END], method="ffill")[0] + 1
    prices_df = full_df.iloc[:data_end_index]
    extra_test_df = full_df.iloc[data_end_index:]

    return prices_df, train_volatilities, eval_volatilities, test_volatilities, extra_test_volatilities, extra_test_df

if __name__ == "__main__":
    prices = load_or_update_price_data(force_refresh=True)
    #print(prices.head())