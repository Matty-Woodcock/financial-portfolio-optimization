import numpy as np

class RewardFunction:
    def __init__(self, volatility_in_observation_space = False, dcc_in_observation_space = False):
        self.volatility_in_observation_space = volatility_in_observation_space
        self.dcc_in_observation_space        = dcc_in_observation_space
        
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
    def __call__(self, portfolio_value_before, portfolio_value_after, weights, daily_volatilities, alpha = 0.3, **kwargs):
        pure_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        daily_volatilities_with_cash = np.concatenate(([0], daily_volatilities))
        portfolio_variance = np.sum((weights ** 2) * (daily_volatilities_with_cash ** 2))
        return pure_return - alpha * portfolio_variance
    
class LowVolatilityDCC(RewardFunction):
    """
    Full DCC-GARCH covariance penalty: wᵀ Σ_t w (replaces diagonal approximation).
    Falls back to the diagonal when daily_covariance is None (e.g. during burn-in).
    """
    def __init__(self, alpha=0.3, volatility_in_observation_space=False, dcc_in_observation_space=False):
        super().__init__(volatility_in_observation_space, dcc_in_observation_space)
        self.alpha = alpha

    def __call__(self, portfolio_value_before, portfolio_value_after, weights,
                 daily_covariance=None, daily_volatilities=None, **kwargs):
        pure_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        n = len(weights)
        if daily_covariance is not None:
            cov_with_cash          = np.zeros((n, n))
            cov_with_cash[1:, 1:]  = daily_covariance   # cash row/col stays zero
            portfolio_variance     = float(weights @ cov_with_cash @ weights)
        elif daily_volatilities is not None:
            vols_with_cash     = np.concatenate(([0], daily_volatilities))
            portfolio_variance = float(np.sum((weights ** 2) * (vols_with_cash ** 2)))
        else:
            portfolio_variance = 0.0
        return pure_return - self.alpha * portfolio_variance


class NGSA2(RewardFunction): #TO DO: Fix name
    def __call__(self, **kwargs):
        raise NotImplementedError #No reward function for NGSA2

