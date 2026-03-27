import requests
import pandas as pd
import os

from datetime import datetime, timedelta
from dotenv import load_dotenv

from config.constants import ALL_SYMBOLS, DATA_START, DATA_END, EXTRA_TEST_END

load_dotenv()
API_KEY = os.getenv("TIINGO_API_KEY")

def fetch_tiingo_price_data_for_symbol(symbol, start_date_dt, end_date_dt):
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
        if data.index.tz is not None:
            data.index = data.index.tz_convert(None)
        return data["adjClose"]
    except ValueError as error:
        raise error
    

def fetch_tiingo_price_data_for_all_symbols():
    try:
        if EXTRA_TEST_END:
            price_df = pd.concat({symbol: fetch_tiingo_price_data_for_symbol(symbol, DATA_START, EXTRA_TEST_END) for symbol in ALL_SYMBOLS}, axis=1)
        else:
            price_df = pd.concat({symbol: fetch_tiingo_price_data_for_symbol(symbol, DATA_START, DATA_END) for symbol in ALL_SYMBOLS}, axis=1)
    except ValueError as error:
        raise error
    
    return price_df

if __name__ == "__main__":
    price_df = fetch_tiingo_price_data_for_all_symbols()
    print(price_df)

#There are approximately 252 trading days in a year!