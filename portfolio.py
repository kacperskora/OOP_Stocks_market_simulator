import torch 

class PORTFOLIO: 
    """
    Klasa reprezentująca portfolio z konkretnymi wagami
    """
    def __init__(self, weights: torch.Tensor): 
        
        if weights.dim() != 1: 
            raise ValueError("Wagi muszą byc jednowymiarowe")
        
        if weights.sum() != 1: 
            raise ValueError('Wagi muszą sumować sie do jedynki')
        
        self.weights = weights.float()
        
    def portfolio_returns(self, returns: torch.Tensor): 
        """
        Funckja zliczająca zyski naszego portfolio
        """
        
        if returns.dim() != 2:
            raise ValueError("zwroty musza byc dwuwymiarowym tensorem o wymiarach (T, N)")

        if returns.shape[1] != self.weights.shape[0]:
            raise ValueError("Błąd wymiaru pomiędzy zwrotami i wagami")

        return returns @ self.weights
    
    
    def expected_return(self, mean_returns: torch.Tensor): 
        """
        Funckja zliczająca przeciętne przewidywane zyski naszego portfolio
        """
        return mean_returns @ self.weights
    
    def variance(self, covariance: torch.Tensor): 
        """
        Funkcja zwracająca wariancję naszego portfela
        """
        
        return self.weights @ covariance @ self.weights
    
    def volatility(self, covariance: torch.Tensor): 
        """
        Funkcja zliczająca zmiennosc naszego portfela (odchylenie standardowe)
        """
        return torch.sqrt(self.variance(covariance))
  