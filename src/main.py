import torch 

from market import Asset, Market
from portfolio import PORTFOLIO
from risk_models import SampleCovariance, EWMA
from simulator import MarketSimulator
from analysis import PerformanceAnalyser
from data_loader import load_data

prices_apple = load_data('AAPL', "2020-01-01", "2024-01-01").flatten()
prices_amazon = load_data('AMZN', "2020-01-01", "2024-01-01").flatten()
prices_lockheed = load_data('LMT', "2020-01-01", "2024-01-01").flatten()
prices_nvidia = load_data('NVDA', "2020-01-01", "2024-01-01").flatten()

a = Asset('AAPL', prices_apple)
b = Asset('AMZN', prices_amazon)
c = Asset('LMT', prices_lockheed)
d = Asset('NVDA', prices_nvidia)


market = Market([a, b, c, d])

ret = market.returns_tensor()
mret = market.mean_returns()
cov_m = market.covariance_matrix()

print(ret)
print(mret)
print(cov_m)



weights = torch.tensor([0.5, 0.2, 0, 0.3])

portfolio = PORTFOLIO(weights)

ert = portfolio.expected_return(mean_returns= mret)
rt = portfolio.portfolio_returns(returns= ret)
vol = portfolio.volatility(cov_m)
var = portfolio.variance(cov_m)

print(ert)
print(rt)
print(vol)
print(var)



sample_model = SampleCovariance()
ewma_model = EWMA()

cov_sample = sample_model.covariance(ret)
print(cov_sample)

cov_ewma = ewma_model.covariance(ret)
print(cov_ewma)


# Symulacje
T = 90
S = 1000
sim = MarketSimulator(mret, cov_m)
simulated = sim.simulate(T, S)
portfolio_simulated = sim.portfolio_returns(portfolio= portfolio, T= T, S = S)




analyzer = PerformanceAnalyser(portfolio_simulated)
print('Przeciętny zwrot:', analyzer.mean_returns().item())
print('Zmienność: ', analyzer.volatility().item())

# Wykresy
analyzer.plot_cumulative_returns()
analyzer.plot_distribution()

