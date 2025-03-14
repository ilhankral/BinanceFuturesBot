from config import TRADING_STRATEGY, INITIAL_TAKE_PROFIT_PCT, INITIAL_STOP_LOSS_PCT
from indicators import calculate_ema, calculate_macd, calculate_rsi

# ✅ Check which strategy is selected
def should_enter_trade(data):
    if TRADING_STRATEGY == "EMA_CROSSOVER":
        return ema_crossover_strategy(data)
    elif TRADING_STRATEGY == "MACD_TREND":
        return macd_trend_strategy(data)
    elif TRADING_STRATEGY == "RSI_OVERBOUGHT_OVERSOLD":
        return rsi_strategy(data)
    else:
        return None, "No valid strategy selected"

# ✅ EMA Crossover Strategy
def ema_crossover_strategy(data):
    data["EMA_50"] = calculate_ema(data, 50)
    data["EMA_200"] = calculate_ema(data, 200)

    latest = data.iloc[-1]
    previous = data.iloc[-2]

    if previous["EMA_50"] < previous["EMA_200"] and latest["EMA_50"] > latest["EMA_200"]:
        return "LONG", "EMA Bullish Crossover"
    elif previous["EMA_50"] > previous["EMA_200"] and latest["EMA_50"] < latest["EMA_200"]:
        return "SHORT", "EMA Bearish Crossover"

    return None, "No EMA signal"

# ✅ MACD Trend Strategy
def macd_trend_strategy(data):
    data["MACD"], data["MACD_signal"] = calculate_macd(data)

    latest = data.iloc[-1]

    if latest["MACD"] > latest["MACD_signal"]:
        return "LONG", "MACD Bullish Trend"
    elif latest["MACD"] < latest["MACD_signal"]:
        return "SHORT", "MACD Bearish Trend"

    return None, "No MACD signal"

# ✅ RSI Overbought/Oversold Strategy
def rsi_strategy(data):
    data["RSI"] = calculate_rsi(data)

    latest = data.iloc[-1]

    if latest["RSI"] < 30:
        return "LONG", "RSI Oversold"
    elif latest["RSI"] > 70:
        return "SHORT", "RSI Overbought"

    return None, "No RSI signal"

# ✅ Check if we should exit a trade
def should_exit_trade(entry_price, current_price, trade_type):
    if trade_type == "LONG":
        profit_pct = (current_price - entry_price) / entry_price
    else:  # SHORT trade handling
        profit_pct = (entry_price - current_price) / entry_price

    # 📌 Take Profit Condition
    if profit_pct >= INITIAL_TAKE_PROFIT_PCT:
        return True, "Take Profit Reached"

    # 📌 Stop-Loss Condition
    elif profit_pct <= INITIAL_STOP_LOSS_PCT:
        return True, "Stop-Loss Triggered"

    return False, "Trade Still Active"
