from config import TRADING_STRATEGIES, INITIAL_TAKE_PROFIT_PCT, INITIAL_STOP_LOSS_PCT
import pandas as pd
import numpy as np
from src.indicators import calculate_ema, calculate_macd, calculate_rsi

def analyze_market_conditions(data):
    """Analyze overall market conditions"""
    # Calculate volatility using ATR
    data['TR'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['ATR'] = data['TR'].rolling(window=14).mean()
    
    # Calculate volume trend
    data['volume_sma'] = data['volume'].rolling(window=20).mean()
    volume_trend = data['volume'].iloc[-1] > data['volume_sma'].iloc[-1]
    
    # Calculate price momentum
    momentum = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
    
    return {
        'volatility': data['ATR'].iloc[-1],
        'volume_trend': volume_trend,
        'momentum': momentum
    }

def should_enter_trade(data):
    # Add OHLCV columns if not present
    if 'open' not in data.columns:
        data['open'] = pd.to_numeric(data['open'])
        data['high'] = pd.to_numeric(data['high'])
        data['low'] = pd.to_numeric(data['low'])
        data['close'] = pd.to_numeric(data['close'])
        data['volume'] = pd.to_numeric(data['volume'])

    # Analyze market conditions
    market_conditions = analyze_market_conditions(data)
    
    signals = []
    
    if "EMA_CROSSOVER" in TRADING_STRATEGIES:
        ema_signal, ema_reason = ema_crossover_strategy(data)
        if ema_signal:
            signals.append((ema_signal, ema_reason))

    if "MACD_TREND" in TRADING_STRATEGIES:
        macd_signal, macd_reason = macd_trend_strategy(data)
        if macd_signal:
            signals.append((macd_signal, macd_reason))

    if "RSI_OVERBOUGHT_OVERSOLD" in TRADING_STRATEGIES:
        rsi_signal, rsi_reason = rsi_strategy(data)
        if rsi_signal:
            signals.append((rsi_signal, rsi_reason))

    # Count signals
    long_signals = sum(1 for s in signals if s[0] == "LONG")
    short_signals = sum(1 for s in signals if s[0] == "SHORT")

    # Additional market condition filters
    if long_signals >= 2:
        # Confirm trend with market conditions
        if (market_conditions['momentum'] > 0 and 
            market_conditions['volume_trend'] and 
            market_conditions['volatility'] < data['close'].iloc[-1] * 0.02):  # Volatility < 2% of price
            return "LONG", f"Multi-Strategy Confirmation: {[s[1] for s in signals if s[0] == 'LONG']}"
    elif short_signals >= 2:
        # Confirm trend with market conditions
        if (market_conditions['momentum'] < 0 and 
            market_conditions['volume_trend'] and 
            market_conditions['volatility'] < data['close'].iloc[-1] * 0.02):
            return "SHORT", f"Multi-Strategy Confirmation: {[s[1] for s in signals if s[0] == 'SHORT']}"

    return None, "No strong confirmation or market conditions unfavorable"

def ema_crossover_strategy(data):
    # Calculate EMAs
    data = calculate_ema(data, 50)
    data = calculate_ema(data, 200)
    
    # Add price action confirmation
    data['higher_highs'] = data['high'] > data['high'].shift(1)
    data['higher_lows'] = data['low'] > data['low'].shift(1)
    data['lower_highs'] = data['high'] < data['high'].shift(1)
    data['lower_lows'] = data['low'] < data['low'].shift(1)

    latest = data.iloc[-1]
    previous = data.iloc[-2]

    # Strong bullish trend
    if (previous["EMA_50"] < previous["EMA_200"] and 
        latest["EMA_50"] > latest["EMA_200"] and 
        latest['higher_highs'] and latest['higher_lows']):
        return "LONG", "EMA Bullish Crossover with Price Action Confirmation"
    
    # Strong bearish trend
    elif (previous["EMA_50"] > previous["EMA_200"] and 
          latest["EMA_50"] < latest["EMA_200"] and 
          latest['lower_highs'] and latest['lower_lows']):
        return "SHORT", "EMA Bearish Crossover with Price Action Confirmation"

    return None, "No EMA signal"

def macd_trend_strategy(data):
    # Calculate MACD
    data = calculate_macd(data)
    
    # Add volume confirmation
    data['volume_sma'] = data['volume'].rolling(window=20).mean()
    volume_trend = data['volume'] > data['volume_sma']

    latest = data.iloc[-1]
    previous = data.iloc[-2]

    # Strong bullish trend
    if (latest["MACD"] > latest["MACD_signal"] and 
        previous["MACD"] <= previous["MACD_signal"] and 
        latest['volume'] > latest['volume_sma']):
        return "LONG", "MACD Bullish Crossover with Volume Confirmation"
    
    # Strong bearish trend
    elif (latest["MACD"] < latest["MACD_signal"] and 
          previous["MACD"] >= previous["MACD_signal"] and 
          latest['volume'] > latest['volume_sma']):
        return "SHORT", "MACD Bearish Crossover with Volume Confirmation"

    return None, "No MACD signal"

def rsi_strategy(data):
    # Calculate RSI
    data = calculate_rsi(data)
    
    # Add trend confirmation
    data['sma_20'] = data['close'].rolling(window=20).mean()
    trend_up = data['close'] > data['sma_20']

    latest = data.iloc[-1]
    previous = data.iloc[-2]

    # Oversold with bullish divergence
    if (latest["RSI"] < 30 and 
        latest["RSI"] > previous["RSI"] and 
        latest['close'] < previous['close'] and 
        trend_up.iloc[-1]):
        return "LONG", "RSI Oversold with Bullish Divergence"
    
    # Overbought with bearish divergence
    elif (latest["RSI"] > 70 and 
          latest["RSI"] < previous["RSI"] and 
          latest['close'] > previous['close'] and 
          not trend_up.iloc[-1]):
        return "SHORT", "RSI Overbought with Bearish Divergence"

    return None, "No RSI signal"

def should_exit_trade(entry_price, current_price, trade_type):
    if trade_type == "LONG":
        profit_pct = (current_price - entry_price) / entry_price
    else:  # SHORT trade
        profit_pct = (entry_price - current_price) / entry_price

    # Dynamic take-profit based on momentum
    if profit_pct >= INITIAL_TAKE_PROFIT_PCT * 1.5:  # 50% more than initial target
        return True, "Extended Take Profit Reached"
    elif profit_pct >= INITIAL_TAKE_PROFIT_PCT:
        return True, "Take Profit Reached"

    # Dynamic stop-loss
    elif profit_pct <= INITIAL_STOP_LOSS_PCT:
        return True, "Stop-Loss Triggered"

    return False, "Trade Still Active"
