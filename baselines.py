import numpy as np

from price_data_reader_writer import load_or_update_price_data
from config.constants import ALL_SYMBOLS_WITH_CASH, GARCH_BURN_IN_DAYS

def main():
    try:
        prices_df = load_or_update_price_data()
    except ValueError as error:
        print(f"Fetching Price Data Failed: {error}")
        return

    burn_in_split_index = GARCH_BURN_IN_DAYS
    eval_split_index = int(len(prices_df[burn_in_split_index:]) * 0.85) + burn_in_split_index
    
    test_df = prices_df.iloc[eval_split_index:]

    test_df_excluding_window_and_burn_in = test_df[39:] #This has been verified to be correct 

    #Choose portfolio weights for baseline
    #["CASH", "AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ", "TLT", "GLD"]
    baseline_weights = [1/len(ALL_SYMBOLS_WITH_CASH)] * len(ALL_SYMBOLS_WITH_CASH)
    #baseline_weights = [0, 0, 0, 0, 0, 1, 0, 0, 0]

    asset_returns = test_df_excluding_window_and_burn_in.pct_change().dropna()
    cash_returns = np.zeros((len(asset_returns), 1))
    asset_returns_with_cash = np.hstack([cash_returns, asset_returns.values])

    portfolio_returns = asset_returns_with_cash @ baseline_weights
    portfolio_values = (1 + portfolio_returns).cumprod()
    print(f"Equal Weight Return = {portfolio_values[-1]}")

if __name__ == "__main__":
    main()