import pandas as pd

# ✅ Exponential Moving Average (EMA)
def calculate_ema(data, period=50):
    return data["close"].ewm(span=period, adjust=False).mean()

# ✅ Relative Strength Index (RSI)
def calculate_rsi(data, period=14):
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ✅ Moving Average Convergence Divergence (MACD)
def calculate_macd(data, fast=12, slow=26, signal=9):
    fast_ema = data["close"].ewm(span=fast, adjust=False).mean()
    slow_ema = data["close"].ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# ✅ Volume Weighted Average Price (VWAP)
def calculate_vwap(data):
    return (data["close"] * data["volume"]).cumsum() / data["volume"].cumsum()