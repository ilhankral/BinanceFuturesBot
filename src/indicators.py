import ta  # ✅ Importing the ta library for indicators

# ✅ Calculate Exponential Moving Average (EMA)
def calculate_ema(df, period):
    df[f"EMA_{period}"] = ta.trend.ema_indicator(df["close"], window=period)
    return df  # 🔹 Return entire DataFrame with EMA column added

# ✅ Calculate MACD & Signal Line
def calculate_macd(df):
    df["MACD"] = ta.trend.macd(df["close"])
    df["MACD_signal"] = ta.trend.macd_signal(df["close"])
    return df  # 🔹 Return DataFrame with MACD columns added

# ✅ Calculate Relative Strength Index (RSI)
def calculate_rsi(df, period=14):
    df["RSI"] = ta.momentum.rsi(df["close"], window=period)
    return df  # 🔹 Return DataFrame with RSI column added

# ✅ Calculate Bollinger Bands (Optional Future Strategy)
def calculate_bollinger_bands(df, period=20, std_dev=2):
    bb = ta.volatility.BollingerBands(df["close"], window=period, window_dev=std_dev)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_middle"] = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()
    return df  # 🔹 Return DataFrame with Bollinger Bands columns added