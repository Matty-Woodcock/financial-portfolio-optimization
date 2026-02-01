import numpy as np

class RewardFunction:
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