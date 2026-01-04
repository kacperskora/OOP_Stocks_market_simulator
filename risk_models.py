import torch 

from abc import ABC, abstractmethod


class RiskModel(ABC): 
    """
    abstrakcyjna klasa dla modeli ryzyka
    """
    
    @abstractmethod
    def covariance(self, returns: torch.Tensor): 
        
        pass
    
    
class SampleCovariance(RiskModel):
    
    def covariance(self, returns: torch.Tensor): 
        
        if returns.dim() != 2:
            raise ValueError('Zwroty muszą byc tensorem dwuwymiarowym o wymiarach (T, N)') 
        
        return torch.cov(returns.T)
    
class EWMA(RiskModel): 
    """
    Exponentially weighted moving average model on returns
    """
    
    def __init__(self, lambda_ : float = 0.94): 
        if not (0.0 < lambda_ < 1): 
            
            raise ValueError("Lambda musi byc z przedzialu 0 do 1")
        
        self.lambda_ = lambda_
    
    def covariance(self, returns: torch.Tensor): 
        
        if returns.dim() != 2:
            raise ValueError('Zwroty muszą byc tensorem dwuwymiarowym o wymiarach (T, N)') 
        
        T, N = returns.shape
        
        mean = returns.mean(dim = 0)
        centered = returns - mean
        
        cov = torch.zeros((N, N), dtype = returns.dtype)
        
        for t in range(T): 
            r_t = centered[t].unsqueeze(1) 
            cov = self.lambda_ * cov +(1 - self.lambda_) * (r_t @ r_t.T)
            
        return cov
        
        