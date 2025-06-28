""" 
Stopping criteria for the MCMC algorithm in the Stochastic Block Model (SBM).
"""

class StoppingCriteria:
    """
    Base class for stopping criteria in the MCMC algorithm.
    """
    def __init__(self, stopping_configs: dict):
        pass

    def should_stop(self, iteration: int, current_ll: float) -> bool:
        """
        Check if the stopping criteria are met.

        :param iteration: Current iteration number.
        :param current_ll: Current log-likelihood value.
        :return: True if the algorithm should stop, False otherwise.
        """
        raise NotImplementedError("Subclasses should implement this method.")