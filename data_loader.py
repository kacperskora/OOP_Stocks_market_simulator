import yfinance as yf
import torch 

def load_data(symbol: str, start: str, end: str): 
    
    data = yf.download(symbol, start, end, progress = False)
    
    if 'Close' not in data.columns: 
        raise ValueError('Adj close is not in data columns')
    
    prices = torch.tensor(data['Close'].values, dtype = torch.float32)
    
    return prices