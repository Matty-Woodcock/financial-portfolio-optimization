import os
import pandas as pd
from datetime import datetime
from price_data_fetcher import fetch_tiingo_price_data_for_all_symbols

CSV_FILE = "price_data.csv"

def load_or_update_price_data(force_refresh=False):
    """
    Loads price data from CSV if it exists, otherwise fetches from Tiingo and saves to CSV.
    """
    if os.path.exists(CSV_FILE) and not force_refresh:
        print(f"Loading price data from CSV...")
        prices_df = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
    else:
        print("Fetching updated price data from Tiingo API...")
        try:
            prices_df = fetch_tiingo_price_data_for_all_symbols()
        except ValueError as error:
            raise error
        if prices_df is not None:
            prices_df.to_csv(CSV_FILE)
            print(f"Price Data Saved To CSV")
    return prices_df

if __name__ == "__main__":
    prices = load_or_update_price_data()
    print(prices.head())