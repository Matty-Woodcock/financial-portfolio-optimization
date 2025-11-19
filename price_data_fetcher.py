import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TIINGO_API_KEY")

def fetch_tiingo_price_data_for_symbol(symbol, start_date_dt, num_weeks = 8):
    interval = timedelta(weeks=num_weeks)
    end_date_dt = start_date_dt + interval
    start_date_str = start_date_dt.strftime("%Y-%m-%d")
    end_date_str = end_date_dt.strftime("%Y-%m-%d")

    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    headers = {"Authorization": f"Token {API_KEY}"}
    params = {"startDate": start_date_str, "endDate": end_date_str}

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
    

def fetch_tiingo_price_data_for_all_symbols():
    all_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ", "TLT", "GLD"]
    start_date = datetime(2018, 6, 1)

    try:
        price_df = pd.concat({symbol: fetch_tiingo_price_data_for_symbol(symbol, start_date, 340) for symbol in all_symbols}, axis=1)
    except ValueError as error:
        print(f"Backtest Failed: {error}")
        return
    
    return price_df

if __name__ == "__main__":
    print(fetch_tiingo_price_data_for_all_symbols())

#240 Weeks: 2020/06/01 - 2025-01-06

#Plan
#2. More data?
#3. Save historic price data