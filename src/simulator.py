import torch 

from market import Market

class MarketSimulator: 
    def __init__(self, mean_returns: torch.Tensor, covariance: torch.Tensor): 
        
        self.mean_returns = mean_returns
        self.covariance = covariance
        self.N = mean_returns.shape[0]
    
    def simulate(self, T: int, S: int): 
        
        distribution = torch.distributions.MultivariateNormal(self.mean_returns, self.covariance)
        simulated_returns = distribution.sample((S, T))
        
        return simulated_returns
    
    
    def portfolio_returns(self, portfolio, T: int, S: int): 
        
        simulated_returns = self.simulate(T, S)
        
        portfolio_returns = simulated_returns @ portfolio.weights
        
        return portfolio_returns
        
        