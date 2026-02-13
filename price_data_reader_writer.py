import os
import numpy as np
import pandas as pd
from datetime import datetime

from price_data_fetcher import fetch_tiingo_price_data_for_all_symbols
from volatility_calculator import volatility_calculator

CSV_FILE = "price_data.csv"

def load_or_update_price_data(force_refresh=False):
    """
    Loads price data from CSV if it exists, otherwise fetches from Tiingo and saves to CSV.
    """
    if os.path.exists(CSV_FILE) and not force_refresh:
        print(f"Loading price data from CSV...")
        prices_df = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
        train_volatilities = np.load("train_volatilities.npy", allow_pickle=True)
        eval_volatilities = np.load("eval_volatilities.npy", allow_pickle=True)
        test_volatilities = np.load("test_volatilities.npy", allow_pickle=True)
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
        np.save("train_volatilities.npy", train_volatilities)
        np.save("eval_volatilities.npy", eval_volatilities)
        np.save("test_volatilities.npy", test_volatilities)

    return prices_df, train_volatilities, eval_volatilities, test_volatilities

if __name__ == "__main__":
    prices = load_or_update_price_data(force_refresh=True)
    #print(prices.head())