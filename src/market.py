import torch 


class Asset: 
    """
    Reprezentuje pojedyncze aktywo i jego ceny w czasie
    """
    def __init__(self, name: str, prices: torch.Tensor): 
        """
        name : str
            Nazwa aktywa (np. 'AAPL')
        prices : torch.Tensor
            Tensor cen o kształcie (T,) T- okres czasu jaki bierzemy
        """
        if prices.dim() != 1: 
            raise ValueError("Cena musi byc tensorem jednowymiarowym")
        
        self.name = name
        self.prices = prices.float()
        
    def returns(self): 
        """
        Oblicza logarytmiczne stopy zwrotu
        
        zwraca tensor o wymiarach (T-1, )
        """
        
        return torch.diff(torch.log(self.prices))
    
class Market: 
    """
    Reprezentuje rynek skladajacy sie z wielu aktywow
    """
    def __init__(self, assets: list[Asset]): 
        
        if len(assets) == 0: 
            raise ValueError('Market musi zawierać jakieś aktywa')
        
        self.assets = assets
        
    def price_tensor(self): 
        """
        Zwraca tensor cen dla wszystkich aktywow
        """
        return torch.stack([asset.prices for asset in self.assets], dim =1)
    
    
    def returns_tensor(self): 
        """
        Zwraca tensor logarytmicznych zwrotow dla wszystkich aktywow
        """
        
        return torch.stack([asset.returns() for asset in self.assets], dim =1)
    
    def mean_returns(self): 
        """
        Zwraca tensor przecietnych zwrotow dla wszystkich aktywow
        """
        
        returns = self.returns_tensor()
        
        return returns.mean(dim = 0)
    
    def covariance_matrix(self): 
        """
        Zwraca macierz kowariancji dla aktywow
        """
        
        returns = self.returns_tensor()
        
        return torch.cov(returns.T)
    
        