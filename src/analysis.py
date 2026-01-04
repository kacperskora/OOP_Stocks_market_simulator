import torch 
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceAnalyser: 
    
    def __init__(self, portfolio_returns: torch.Tensor): 
        
        if portfolio_returns.dim() != 2:
            raise ValueError("portfolio_returns must be a 2D tensor (S, T)")

        self.returns = portfolio_returns
        self.S, self.T = portfolio_returns.shape
        
        
    def cumulative_returns(self): 
        
        return torch.cumprod(1 + self.returns, dim = 1)
    
    def mean_returns(self): 
        
        return self.returns.mean()
    
    def volatility(self): 
        
        return self.returns.std()
    
    
    def plot_cumulative_returns(self):
        """
        Wykres skumulowanych zyskow dla wszystkich scewnariuszy
        """
        cumulative = self.cumulative_returns()
        plt.figure(figsize=(10, 6))
        for i in range(self.S):
            plt.plot(cumulative[i].numpy(), alpha=0.7)
        plt.title("Skumulowane Zyski Portfolio")
        plt.xlabel("Okres Czasu")
        plt.ylabel("Skumulowany Zysk")
        plt.grid(True)
        plt.show()
    
    
    def plot_distribution(self, time_step= -1):
        """
        Histogram dla zwrotow w kazdym z okresow czasu
        """
        data = self.cumulative_returns()[:, time_step]
        plt.figure(figsize=(8, 5))
        sns.histplot(data.numpy(), bins=15, alpha=0.7, kde = True, kde_kws = {'bw_adjust' : 1.5})
        plt.title(f"Skumulowane zwroty portfolio w ostatnim dniu")
        plt.xlabel("Skumulowany Zysk")
        plt.ylabel("Częstość")
        plt.grid(True)
        plt.show()
        