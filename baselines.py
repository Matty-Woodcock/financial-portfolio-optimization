import numpy as np

from price_data_reader_writer import load_or_update_price_data

def main():
    try:
        prices_df = load_or_update_price_data()
    except ValueError as error:
        print(f"Fetching Price Data Failed: {error}")
        return

    eval_split_index = int(len(prices_df) * 0.85)

    test_df = prices_df.iloc[eval_split_index:]

    test_df_excluding_window = test_df[39:] #This has been verified to be correct 

    #Choose portfolio weights for baseline
    #["CASH", "AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ", "TLT", "GLD"]
    baseline_weights = [1/9] * 9
    #baseline_weights = [0, 0, 0, 0, 0, 0, 0, 0, 1]

    asset_returns = test_df_excluding_window.pct_change().dropna()
    cash_returns = np.zeros((len(asset_returns), 1))
    asset_returns_with_cash = np.hstack([cash_returns, asset_returns.values])

    portfolio_returns = asset_returns_with_cash @ baseline_weights
    portfolio_value = (1 + portfolio_returns).cumprod()
    print(portfolio_value)

if __name__ == "__main__":
    main()

#[np.float64(1.2046245654012437), np.float32(0.21844052), np.float64(-0.10799657863516023)]