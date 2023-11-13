import QuantLib as ql
import yfinance as yf
import numpy as np

# Define the stock symbol and the date range
stock_symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2022-12-31"

# Download historical stock data from Yahoo Finance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Calculate daily returns
stock_data['Returns'] = stock_data['Adj Close'].pct_change().dropna()

# Convert returns to a NumPy array
returns = stock_data['Returns'].values

# Initialize QuantLib objects for the Heston model
calculation_date = ql.Date(1, 1, 1)
spot_price = stock_data['Adj Close'].iloc[-1]
day_count = ql.Actual360()
calendar = ql.UnitedStates()
calculation_date = calendar.adjust(calculation_date)

# Define risk-free rate, dividend rate, and yield term structure
risk_free_rate = 0.01
dividend_rate = 0.0
yts = ql.FlatForward(calculation_date, ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)), day_count)
yts.enableExtrapolation()

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
flat_ts = ql.YieldTermStructureHandle(yts)
dividend_handle = ql.QuoteHandle(ql.SimpleQuote(dividend_rate))

# Define Heston model parameters, hypertune them for higher accuracy using machine learning.
v0 = 0.1
kappa = 0.1
theta = 0.1
sigma = 0.1
rho = -0.75

# Create Heston process and model
process = ql.HestonProcess(flat_ts, dividend_handle, spot_handle, v0, kappa, theta, sigma, rho)
model = ql.HestonModel(process)
engine = ql.AnalyticHestonEngine(model)

# Create a vanilla European call option
heston_option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, spot_price), ql.EuropeanExercise(ql.Date(1, 1, 1)))
heston_option.setPricingEngine(engine)

# Calculate implied volatility using the Heston model
annual_volatility = heston_option.impliedVolatility(returns[-1], ql.Settings(), 1.0e-10, 1000, 0.0, 4.0)

# Print the results
print(f"Highly Stock Price Volatility Analysis for {stock_symbol}")
print(f"From: {start_date}")
print(f"To: {end_date}")
print(f"Annualized Volatility (Heston Model): {annual_volatility:.6f}")
