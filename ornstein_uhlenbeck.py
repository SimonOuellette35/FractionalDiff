import numpy.random as nrand

class ModelParameters:
    """
    Encapsulates model parameters
    """
    def __init__(self, all_time, starting_level, theta, mean, trend):
        # This is the amount of time to simulate for
        self.all_time = all_time
        self.starting_level = starting_level
        self.theta = theta                      # strength of mean reversion
        self.mean = mean                        # long-term mean to which the process reverts
        self.trend = trend                      # amount by which the long-term mean increases at every dt

def brownian_motion_log_returns(param):
    return nrand.normal(loc=0, scale=1, size=param.all_time)

def ornstein_uhlenbeck_levels(param):
    # dx(t) = theta((mu + trend) - x(t)) dt + sigma * dW(t), where theta > 0, mu and sigma > 0
    ou_levels = [param.starting_level]
    brownian_motion_returns = brownian_motion_log_returns(param)
    mean_trend = param.mean
    for i in range(1, param.all_time):
        drift = param.theta * (mean_trend - ou_levels[i-1])
        randomness = brownian_motion_returns[i - 1]
        ou_levels.append(ou_levels[i - 1] + drift + randomness)
        mean_trend += param.trend

    return ou_levels