import gymnasium as gym
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TIINGO_API_KEY")

def get_tiingo_price_data(symbol, start_date, end_date):
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    headers = {"Authorization": f"Token {API_KEY}"}
    params = {"startDate": start_date, "endDate": end_date}

    try:
        response = requests.get(url, headers=headers, params=params)
        json_data = response.json()
        if isinstance(json_data, dict) and "detail" in json_data:
                raise ValueError(f"Tiingo API error: {json_data['detail']}")
    
        data = pd.DataFrame(response.json())
        data["date"] = pd.to_datetime(data["date"])
        data.set_index("date", inplace=True)
        return data["adjClose"]
    except ValueError as error:
        raise error


def test_portfolio_one_week(portfolio, current_backtest_date, interval=timedelta(weeks=8)):
    after_interval_backtest_date = current_backtest_date + interval
    request_start_date = current_backtest_date.strftime("%Y-%m-%d")
    request_end_date = after_interval_backtest_date.strftime("%Y-%m-%d")

    try:
        prices = pd.concat({symbol: get_tiingo_price_data(symbol, request_start_date, request_end_date) for symbol in portfolio}, axis=1)
    except ValueError as error:
        raise error
    
    weights = list(portfolio.values())
    prices.columns = list(portfolio.keys())
    #print(prices)

    weekly_groups = []
    for _, week_price_data in prices.groupby(pd.Grouper(freq="W-MON", label="left", closed="left")):
        if not week_price_data.empty:
            week_returns_data = week_price_data.pct_change().dropna()
            weekly_portfolio_return = (week_returns_data * weights).sum(axis=1)
            weekly_portfolio_value = (1 + weekly_portfolio_return).cumprod().tail(1)
            weekly_groups.append((weekly_portfolio_value, week_price_data))

    return weekly_groups


def run_backtest():
    starting_backtest_portfolio = {"AAPL": 0.4, "MSFT": 0.4, "GOOGL": 0.2}
    current_backtest_date = datetime(2024, 6, 1)

    try:
        backtest_weekly_groups = test_portfolio_one_week(starting_backtest_portfolio, current_backtest_date)
    except ValueError as error:
        print(f"Backtest Failed: {error}")
        return
    
    weekly_values = pd.concat([val for val, _ in backtest_weekly_groups])
    print(weekly_values)

    plt.figure(figsize=(10, 5))
    plt.plot(weekly_values)
    plt.title("Cumulative Portfolio Return Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_backtest()

#Plan
#3. Save historic price data