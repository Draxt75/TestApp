from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from scipy.stats import norm

app = FastAPI()

class BlackScholesInput(BaseModel):
    stockPrice: float
    strikePrice: float
    timeToExpiry: float
    riskFreeRate: float
    volatility: float

def calculate_black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call

def calculate_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}

@app.post("/black-scholes")
async def black_scholes(input: BlackScholesInput):
    price = calculate_black_scholes(input.stockPrice, input.strikePrice, input.timeToExpiry, input.riskFreeRate, input.volatility)
    greeks = calculate_greeks(input.stockPrice, input.strikePrice, input.timeToExpiry, input.riskFreeRate, input.volatility)
    return {"price": price, "greeks": greeks}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)