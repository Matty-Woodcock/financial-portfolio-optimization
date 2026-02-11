# from arch import arch_model

# class GARCHVolatilityModel:
#     def __init__(self, asset_returns):
#         self.asset_returns = asset_returns

#     def calculate_volatility(self):
#         returns = asset_returns['AAPL']  # daily returns
#         print(self.asset_returns["AAPL"])
#         # model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
#         # res = model.fit(disp='off')

#         # # Get conditional volatility (sigma_t)
#         # sigma_t = res.conditional_volatility

import numpy as np

from collections import deque
from arch import arch_model

class GARCHVolatilityModel:
    def __init__(self, window_size=250, p=1, q=1, refit_every=5):
        self.window_size = window_size
        self.p = p
        self.q = q
        self.refit_every = refit_every
        
        self.returns_buffer = deque(maxlen=window_size)
        self.steps_since_refit = 0
        self.model_fit = None  # last fitted GARCH model
        self.last_forecast = None  # last forecasted variance

        self.total_steps = 0
    
    def add_return(self, asset_return):
        self.returns_buffer.append(asset_return * 100)
        self.steps_since_refit += 1
        self.total_steps += 1

    def can_forecast(self):
        return len(self.returns_buffer) == self.window_size

    def forecast_volatility(self):
        if not self.can_forecast():
            return None #Not enough data to forecast probability

        #Refit GARCH only every refit_every steps
        if (self.model_fit is None) or (self.steps_since_refit >= self.refit_every):
            model = arch_model(np.array(self.returns_buffer), vol='Garch', p=self.p, q=self.q, dist='normal', rescale=False)
            self.model_fit = model.fit(disp='off')
            self.steps_since_refit = 0

        #Forecast variance of next step
        forecast = self.model_fit.forecast(horizon=1)
        self.last_forecast = np.sqrt(forecast.variance.values[-1, 0])
        return self.last_forecast / 100