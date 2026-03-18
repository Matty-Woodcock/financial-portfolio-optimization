from datetime import datetime

ALL_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ", "TLT", "GLD"]
#ALL_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ", "TLT", "GLD", "KO", "BA", "XOM"]
ALL_SYMBOLS_WITH_CASH = ["CASH"] + ALL_SYMBOLS

GARCH_BURN_IN_DAYS = 250

TRAIN_SPLIT = 0.70
EVAL_SPLIT = 0.80
RL_WINDOW_SIZE = 40
DATA_START = datetime(2012, 6, 2) #datetime(2017, 6, 2)
DATA_END = datetime(2025, 12, 6) #datetime(2024, 12, 6)