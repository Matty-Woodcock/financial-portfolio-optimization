import pandas as pd
import numpy as np
import os

def read_rf_rates():
    csv_path = os.path.join(os.path.dirname(__file__), "rf_rates.csv")
    df = pd.read_csv(csv_path, parse_dates = ["observation_date"])
    df = df.set_index("observation_date")
    df["daily_rf"] = (1 + df["TB3MS"] / 100) ** (1 / 252) - 1

    #Convert monthly observations to a daily series by forward-filling
    df = df.resample("B").ffill()  #"B" = business days
    return df["daily_rf"]

RF_DAILY_SERIES = read_rf_rates()

def get_daily_rf(date): #Look up the daily risk-free rate for a given date
    if hasattr(date, 'tzinfo') and date.tzinfo is not None:
        date = date.replace(tzinfo=None)
        
    try:
        return RF_DAILY_SERIES[date]
    except KeyError:
        nearest_index = RF_DAILY_SERIES.index.get_indexer([date], method="nearest")[0]
        return RF_DAILY_SERIES.iloc[nearest_index]