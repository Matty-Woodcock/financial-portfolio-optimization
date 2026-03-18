import numpy as np

class RewardFunction:
    def __init__(self, volatility_in_observation_space = False):
        self.volatility_in_observation_space = volatility_in_observation_space
        
    def __call__(self, **kwargs):
        raise NotImplementedError
    
class PureReturn(RewardFunction):
    def __call__(self, portfolio_value_before, portfolio_value_after, **kwargs):
        pure_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        return pure_return
    
class LogReturn(RewardFunction):
    def __call__(self, portfolio_value_before, portfolio_value_after, **kwargs):
        log_return = np.log(portfolio_value_after / (portfolio_value_before))
        return log_return
    
class HighHHI(RewardFunction):
    def __call__(self, portfolio_value_before, portfolio_value_after, weights, alpha = 0.15, **kwargs):
        pure_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        return pure_return + alpha * np.sum(np.square(weights))
    
class LowVolatility(RewardFunction):
    def __call__(self, portfolio_value_before, portfolio_value_after, weights, daily_volatilities, alpha = 0.15, **kwargs):
        pure_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        daily_volatilities_with_cash = np.concatenate(([0], daily_volatilities))
        portfolio_variance = np.sum((weights ** 2) * (daily_volatilities_with_cash ** 2))
        return pure_return - alpha * portfolio_variance
    
class NGSA2(RewardFunction):
    def __call__(self, **kwargs):
        raise NotImplementedError #No reward function for NGSA2

