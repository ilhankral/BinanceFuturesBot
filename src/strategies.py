"""
Enhanced Trading Strategies Module
- Multi-strategy confirmation system
- Market regime detection
- Adaptive entry/exit conditions
- Signal scoring system
- Comprehensive market condition analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import ta
from scipy import stats

# Import indicator functions
from src.indicators import (
    calculate_ema, calculate_macd, calculate_rsi, calculate_bollinger_bands,
    calculate_atr, calculate_supertrend, calculate_ichimoku, calculate_volume_profile,
    calculate_moving_averages, calculate_momentum_indicators
)

# Import configuration
from src.config import (
    TRADING_STRATEGIES, STRATEGY_WEIGHTS, STRATEGY_PARAMS,
    MIN_STRATEGY_CONFIRMATION, SIGNAL_THRESHOLD
)


def analyze_market_conditions(data: pd.DataFrame) -> Dict:
    """
    Comprehensive market condition analysis

    Returns:
        Dict with market condition metrics
    """
    if data is None or len(data) < 50:
        return {
            "volatility": 0,
            "volume_trend": False,
            "trend_strength": 0,
            "momentum": 0,
            "support_resistance": [],
            "liquidity": 0,
            "market_sentiment": "neutral"
        }

    try:
        # Calculate ATR for volatility
        data = calculate_atr(data, period=14)
        current_atr = data['ATR'].iloc[-1]
        normalized_atr = current_atr / data['close'].iloc[-1]  # Normalize by price

        # Calculate volume trend
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        volume_trend = data['volume'].iloc[-1] > data['volume_sma'].iloc[-1]

        # Calculate volume deviation (for abnormal volume detection)
        data['volume_std'] = data['volume'].rolling(window=20).std()
        volume_z_score = (data['volume'].iloc[-1] - data['volume_sma'].iloc[-1]) / data['volume_std'].iloc[-1]
        abnormal_volume = abs(volume_z_score) > 2.0

        # Calculate ADX for trend strength
        data['plus_dm'] = data['high'].diff().clip(lower=0)
        data['minus_dm'] = (-data['low'].diff()).clip(lower=0)
        data['plus_di'] = 100 * ta.trend.ema_indicator(data['plus_dm'], 14) / data['ATR']
        data['minus_di'] = 100 * ta.trend.ema_indicator(data['minus_dm'], 14) / data['ATR']
        data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
        data['ADX'] = ta.trend.ema_indicator(data['dx'], 14)
        trend_strength = data['ADX'].iloc[-1]

        # Calculate momentum
        momentum = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]

        # Find support/resistance levels using recent highs and lows
        close = data['close'].iloc[-1]
        recent_highs = data['high'].iloc[-30:].nlargest(5).tolist()
        recent_lows = data['low'].iloc[-30:].nsmallest(5).tolist()

        # Filter support/resistance levels by proximity
        support_levels = [level for level in recent_lows if level < close]
        resistance_levels = [level for level in recent_highs if level > close]

        # Cluster nearby levels
        def cluster_levels(levels, threshold=0.01):
            if not levels:
                return []

            clustered = []
            current_cluster = [levels[0]]

            for level in levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold:
                    current_cluster.append(level)
                else:
                    clustered.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [level]

            if current_cluster:
                clustered.append(sum(current_cluster) / len(current_cluster))

            return clustered

        support_resistance = {
            "support": sorted(cluster_levels(support_levels)),
            "resistance": sorted(cluster_levels(resistance_levels))
        }

        # Calculate market sentiment based on multiple indicators
        rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
        macd_signal = 1 if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1] else -1 if data['MACD'].iloc[-1] < \
                                                                                           data['MACD_signal'].iloc[
                                                                                               -1] else 0 if 'MACD' in data.columns and 'MACD_signal' in data.columns else 0
        ema_trend = 1 if data['close'].iloc[-1] > data['EMA_50'].iloc[-1] else -1 if 'EMA_50' in data.columns else 0

        # Combine sentiment indicators
        sentiment_score = 0
        if rsi > 70:
            sentiment_score -= 2  # Overbought
        elif rsi < 30:
            sentiment_score += 2  # Oversold
        else:
            sentiment_score += (rsi - 50) / 10  # Scaled between -2 and 2

        sentiment_score += macd_signal * 2  # MACD signal
        sentiment_score += ema_trend  # EMA trend

        if sentiment_score > 3:
            market_sentiment = "strongly_bullish"
        elif sentiment_score > 1:
            market_sentiment = "bullish"
        elif sentiment_score < -3:
            market_sentiment = "strongly_bearish"
        elif sentiment_score < -1:
            market_sentiment = "bearish"
        else:
            market_sentiment = "neutral"

        # Calculate market liquidity metric using bid-ask spreads (if available)
        bid_ask_spread = data.get('spread', pd.Series([0.001] * len(data))).iloc[-20:].mean()
        normalized_spread = bid_ask_spread / data['close'].iloc[-1]
        liquidity = 1 - min(1, normalized_spread * 1000)  # Higher is more liquid

        return {
            "volatility": normalized_atr,
            "atr_value": current_atr,
            "volume_trend": volume_trend,
            "abnormal_volume": abnormal_volume,
            "volume_z_score": volume_z_score,
            "trend_strength": trend_strength,
            "momentum": momentum,
            "support_resistance": support_resistance,
            "liquidity": liquidity,
            "market_sentiment": market_sentiment,
            "rsi": rsi,
            "macd_signal": macd_signal
        }

    except Exception as e:
        print(f"Error in market conditions analysis: {e}")
        return {
            "volatility": 0,
            "volume_trend": False,
            "trend_strength": 0,
            "momentum": 0,
            "support_resistance": {"support": [], "resistance": []},
            "liquidity": 0,
            "market_sentiment": "neutral"
        }


def detect_market_regime(data: pd.DataFrame) -> str:
    """
    Detect current market regime (trending, ranging, volatile, or choppy)

    Returns:
        str: Market regime type
    """
    if data is None or len(data) < 50:
        return "unknown"

    try:
        # Add indicators if not already present
        if 'ATR' not in data.columns:
            data = calculate_atr(data)

        if 'ADX' not in data.columns:
            # Calculate ADX (Average Directional Index) for trend strength
            data['plus_dm'] = data['high'].diff().clip(lower=0)
            data['minus_dm'] = (-data['low'].diff()).clip(lower=0)
            data['plus_di'] = 100 * ta.trend.ema_indicator(data['plus_dm'], 14) / data['ATR']
            data['minus_di'] = 100 * ta.trend.ema_indicator(data['minus_dm'], 14) / data['ATR']
            data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
            data['ADX'] = ta.trend.ema_indicator(data['dx'], 14)

        # Calculate Bollinger Bands Width for volatility
        if 'BB_width' not in data.columns:
            data = calculate_bollinger_bands(data)
            data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']

        # Get recent values
        adx = data['ADX'].iloc[-1]
        bb_width = data['BB_width'].iloc[-1]
        atr_pct = data['ATR'].iloc[-1] / data['close'].iloc[-1] * 100  # ATR as percentage of price

        # Compute price direction consistency (recent up/down candles)
        last_candles = data.iloc[-10:]
        up_candles = sum(
            1 for i in range(len(last_candles)) if last_candles['close'].iloc[i] > last_candles['open'].iloc[i])
        down_candles = sum(
            1 for i in range(len(last_candles)) if last_candles['close'].iloc[i] < last_candles['open'].iloc[i])
        direction_consistency = max(up_candles, down_candles) / 10  # 0.5 = random, 1.0 = all same direction

        # Determine market regime
        if adx > 25 and direction_consistency > 0.7:
            # Strong trend
            return "trending"
        elif bb_width > 0.05 and atr_pct > 3.0:
            # High volatility
            return "volatile"
        elif adx < 20 and bb_width < 0.03:
            # Low volatility, no trend
            return "ranging"
        else:
            # Mixed signals
            return "choppy"

    except Exception as e:
        print(f"Error detecting market regime: {e}")
        return "unknown"


def get_strategy_combination_score(signals: List[Tuple[str, str]],
                                   market_conditions: Dict,
                                   market_regime: str) -> float:
    """
    Calculate a combined signal strength score (0-100) based on:
    - Number of confirming signals
    - Signal weights
    - Signal agreement
    - Market conditions

    Returns:
        float: Signal score from 0 to 100
    """
    if not signals:
        return 0.0

    try:
        # Count signals by direction
        long_signals = [s for s in signals if s[0] == "LONG"]
        short_signals = [s for s in signals if s[0] == "SHORT"]

        # Determine signal direction based on majority
        signal_direction = "LONG" if len(long_signals) > len(short_signals) else "SHORT"

        # Get confirming signals (those matching the majority direction)
        confirming_signals = long_signals if signal_direction == "LONG" else short_signals
        opposing_signals = short_signals if signal_direction == "LONG" else long_signals

        # Calculate confirmation ratio
        confirmation_ratio = len(confirming_signals) / len(signals) if signals else 0

        # Calculate weighted signal score
        total_weight = 0
        weighted_score = 0

        for signal, reason in confirming_signals:
            # Extract strategy name from reason
            strategy_name = None
            for potential_strategy in TRADING_STRATEGIES:
                if potential_strategy in reason:
                    strategy_name = potential_strategy
                    break

            if not strategy_name:
                # Use a default weight if strategy not identified
                strategy_weight = 1.0
            else:
                # Get strategy weight
                strategy_weight = STRATEGY_WEIGHTS.get(strategy_name, 1.0)

            total_weight += strategy_weight
            weighted_score += strategy_weight

        # Subtract opposing signals weight
        for signal, reason in opposing_signals:
            strategy_name = None
            for potential_strategy in TRADING_STRATEGIES:
                if potential_strategy in reason:
                    strategy_name = potential_strategy
                    break

            if not strategy_name:
                strategy_weight = 1.0
            else:
                strategy_weight = STRATEGY_WEIGHTS.get(strategy_name, 1.0)

            total_weight += strategy_weight
            weighted_score -= strategy_weight * 0.5  # Opposing signals have less negative impact

        if total_weight == 0:
            normalized_signal_score = 0
        else:
            normalized_signal_score = (weighted_score / total_weight) * 100

        # Adjust score based on market conditions
        condition_multiplier = 1.0

        # Adjust for market regime
        if signal_direction == "LONG":
            if market_regime == "trending" and market_conditions.get("momentum", 0) > 0:
                # Trending upward - good for longs
                condition_multiplier *= 1.2
            elif market_regime == "volatile":
                # Volatile - reduce confidence
                condition_multiplier *= 0.8
            elif market_regime == "ranging" and market_conditions.get("rsi", 50) < 40:
                # Ranging and oversold - good for longs
                condition_multiplier *= 1.1

        else:  # SHORT signal
            if market_regime == "trending" and market_conditions.get("momentum", 0) < 0:
                # Trending downward - good for shorts
                condition_multiplier *= 1.2
            elif market_regime == "volatile":
                # Volatile - reduce confidence
                condition_multiplier *= 0.8
            elif market_regime == "ranging" and market_conditions.get("rsi", 50) > 60:
                # Ranging and overbought - good for shorts
                condition_multiplier *= 1.1

        # Adjust for overall market sentiment
        sentiment = market_conditions.get("market_sentiment", "neutral")
        if signal_direction == "LONG" and sentiment in ["bullish", "strongly_bullish"]:
            condition_multiplier *= 1.2
        elif signal_direction == "LONG" and sentiment in ["bearish", "strongly_bearish"]:
            condition_multiplier *= 0.8
        elif signal_direction == "SHORT" and sentiment in ["bearish", "strongly_bearish"]:
            condition_multiplier *= 1.2
        elif signal_direction == "SHORT" and sentiment in ["bullish", "strongly_bullish"]:
            condition_multiplier *= 0.8

        # Calculate final score
        final_score = normalized_signal_score * condition_multiplier

        # Ensure score is between 0 and 100
        final_score = max(0, min(100, final_score))

        return final_score

    except Exception as e:
        print(f"Error calculating strategy score: {e}")
        return 0.0


def should_enter_trade(df_main: pd.DataFrame,
                       df_short: Optional[pd.DataFrame] = None,
                       df_long: Optional[pd.DataFrame] = None,
                       market_conditions: Optional[Dict] = None,
                       market_regime: Optional[str] = None) -> Tuple[Optional[str], str, float]:
    """
    Determine if a trade should be entered based on multiple strategies
    and market conditions.

    Returns:
        Tuple of (signal, reason, score)
    """
    # Add OHLCV columns if not present
    if df_main is None or len(df_main) < 200:  # Need enough data for indicators
        return None, "Insufficient data", 0.0

    try:
        # Ensure data is numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_main.columns:
                df_main[col] = pd.to_numeric(df_main[col])

        # Get market conditions if not provided
        if market_conditions is None:
            market_conditions = analyze_market_conditions(df_main)

        # Get market regime if not provided
        if market_regime is None:
            market_regime = detect_market_regime(df_main)

        # Calculate all required indicators
        df_main = calculate_moving_averages(df_main)
        df_main = calculate_momentum_indicators(df_main)
        df_main = calculate_bollinger_bands(df_main)
        df_main = calculate_atr(df_main)

        # Calculate indicators on additional timeframes if provided
        if df_short is not None:
            df_short = calculate_moving_averages(df_short)
            df_short = calculate_momentum_indicators(df_short)

        if df_long is not None:
            df_long = calculate_moving_averages(df_long)
            df_long = calculate_momentum_indicators(df_long)

        # Collect signals from all strategies
        signals = []

        # Apply each enabled strategy
        for strategy in TRADING_STRATEGIES:
            if strategy == "EMA_CROSSOVER":
                signal, reason = ema_crossover_strategy(df_main)
                if signal:
                    signals.append((signal, reason))

            elif strategy == "MACD_TREND":
                signal, reason = macd_trend_strategy(df_main)
                if signal:
                    signals.append((signal, reason))

            elif strategy == "RSI_OVERBOUGHT_OVERSOLD":
                signal, reason = rsi_strategy(df_main)
                if signal:
                    signals.append((signal, reason))

            elif strategy == "BOLLINGER_BANDS":
                signal, reason = bollinger_strategy(df_main)
                if signal:
                    signals.append((signal, reason))

            elif strategy == "SUPERTREND":
                signal, reason = supertrend_strategy(df_main)
                if signal:
                    signals.append((signal, reason))

            elif strategy == "MULTI_TIMEFRAME":
                # Only use if we have multiple timeframes
                if df_short is not None and df_long is not None:
                    signal, reason = multi_timeframe_strategy(df_main, df_short, df_long)
                    if signal:
                        signals.append((signal, reason))

        # Count long and short signals
        long_signals = sum(1 for s in signals if s[0] == "LONG")
        short_signals = sum(1 for s in signals if s[0] == "SHORT")

        # Calculate signal score
        signal_score = get_strategy_combination_score(signals, market_conditions, market_regime)

        # Determine final signal based on strategy confirmation and score
        if long_signals >= MIN_STRATEGY_CONFIRMATION and long_signals > short_signals:
            reasons = [s[1] for s in signals if s[0] == "LONG"]
            reason_text = f"LONG: Score {signal_score:.1f}/100 - " + ", ".join(reasons[:3])
            return "LONG", reason_text, signal_score

        elif short_signals >= MIN_STRATEGY_CONFIRMATION and short_signals > long_signals:
            reasons = [s[1] for s in signals if s[0] == "SHORT"]
            reason_text = f"SHORT: Score {signal_score:.1f}/100 - " + ", ".join(reasons[:3])
            return "SHORT", reason_text, signal_score

        return None, f"No consensus signal (L:{long_signals}/S:{short_signals}) - Score: {signal_score:.1f}/100", signal_score

    except Exception as e:
        import traceback
        print(f"Error in should_enter_trade: {e}")
        print(traceback.format_exc())
        return None, f"Error: {str(e)}", 0.0


def should_exit_trade(entry_price: float, current_price: float, trade_type: str,
                      market_conditions: Optional[Dict] = None,
                      market_regime: Optional[str] = None,
                      profit_pct: Optional[float] = None,
                      position_age: Optional[timedelta] = None) -> Tuple[bool, str]:
    """
    Determine if a trade should be exited based on price action, profit,
    and market conditions.

    Returns:
        Tuple of (exit_signal, reason)
    """
    # Calculate profit percentage if not provided
    if profit_pct is None:
        if trade_type == "LONG":
            profit_pct = (current_price - entry_price) / entry_price
        else:  # SHORT trade
            profit_pct = (entry_price - current_price) / entry_price

    try:
        # Basic profit targets
        take_profit_threshold = 0.04  # 4% profit
        stop_loss_threshold = -0.02  # 2% loss

        # Adjust targets based on market conditions if provided
        if market_conditions:
            volatility = market_conditions.get("volatility", 0)

            # Scale profit targets with volatility
            volatility_adjustment = min(1.5, max(0.5, volatility * 50))
            take_profit_threshold *= volatility_adjustment
            stop_loss_threshold *= volatility_adjustment

        # Check for take profit
        if profit_pct >= take_profit_threshold:
            return True, f"Take Profit: {profit_pct:.2%}"

        # Check for stop loss
        if profit_pct <= stop_loss_threshold:
            return True, f"Stop Loss: {profit_pct:.2%}"

        # Check for trend reversal if we have market conditions
        if market_conditions and profit_pct > 0:
            sentiment = market_conditions.get("market_sentiment", "neutral")

            # Exit LONG positions in bearish conditions after some profit
            if trade_type == "LONG" and sentiment in ["bearish", "strongly_bearish"] and profit_pct > 0.01:
                return True, f"Market sentiment turned bearish, securing {profit_pct:.2%} profit"

            # Exit SHORT positions in bullish conditions after some profit
            if trade_type == "SHORT" and sentiment in ["bullish", "strongly_bullish"] and profit_pct > 0.01:
                return True, f"Market sentiment turned bullish, securing {profit_pct:.2%} profit"

        # Check for extended time in trade
        if position_age and position_age > timedelta(days=5) and profit_pct > 0:
            return True, f"Extended time in position ({position_age.days} days), securing {profit_pct:.2%} profit"

        # Still in valid trade
        return False, "Trade still active"

    except Exception as e:
        print(f"Error in should_exit_trade: {e}")
        # Exit on error to be safe
        return True, f"Error in exit evaluation: {str(e)}"


def ema_crossover_strategy(data: pd.DataFrame) -> Tuple[Optional[str], str]:
    """Enhanced EMA crossover strategy with price action confirmation"""
    try:
        # Get parameters
        fast_period = STRATEGY_PARAMS.get("EMA_CROSSOVER", {}).get("fast_period", 50)
        slow_period = STRATEGY_PARAMS.get("EMA_CROSSOVER", {}).get("slow_period", 200)

        # Calculate EMAs if not already present
        if f'EMA_{fast_period}' not in data.columns:
            data = calculate_ema(data, fast_period)
        if f'EMA_{slow_period}' not in data.columns:
            data = calculate_ema(data, slow_period)

        # Add price action confirmation
        data['higher_highs'] = (data['high'] > data['high'].shift(1)) & (data['high'].shift(1) > data['high'].shift(2))
        data['higher_lows'] = (data['low'] > data['low'].shift(1)) & (data['low'].shift(1) > data['low'].shift(2))
        data['lower_highs'] = (data['high'] < data['high'].shift(1)) & (data['high'].shift(1) < data['high'].shift(2))
        data['lower_lows'] = (data['low'] < data['low'].shift(1)) & (data['low'].shift(1) < data['low'].shift(2))

        # Add volume confirmation
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        volume_increasing = data['volume'].iloc[-1] > data['volume_sma'].iloc[-1]

        latest = data.iloc[-1]
        previous = data.iloc[-2]

        # Strong bullish trend with confirmation
        if (previous[f"EMA_{fast_period}"] <= previous[f"EMA_{slow_period}"] and
                latest[f"EMA_{fast_period}"] > latest[f"EMA_{slow_period}"]):

            # Add price action and volume confirmations
            confirmation_points = 0
            if latest['higher_highs']:
                confirmation_points += 1
            if latest['higher_lows']:
                confirmation_points += 1
            if volume_increasing:
                confirmation_points += 1

            if confirmation_points >= 2:
                return "LONG", f"EMA Bullish Crossover with {confirmation_points} confirmations"
            else:
                return "LONG", "EMA Bullish Crossover (weak)"

        # Strong bearish trend with confirmation
        elif (previous[f"EMA_{fast_period}"] >= previous[f"EMA_{slow_period}"] and
              latest[f"EMA_{fast_period}"] < latest[f"EMA_{slow_period}"]):

            # Add price action and volume confirmations
            confirmation_points = 0
            if latest['lower_highs']:
                confirmation_points += 1
            if latest['lower_lows']:
                confirmation_points += 1
            if volume_increasing:
                confirmation_points += 1

            if confirmation_points >= 2:
                return "SHORT", f"EMA Bearish Crossover with {confirmation_points} confirmations"
            else:
                return "SHORT", "EMA Bearish Crossover (weak)"

        return None, "No EMA signal"

    except Exception as e:
        print(f"Error in EMA crossover strategy: {e}")
        return None, f"Error: {str(e)}"


def macd_trend_strategy(data: pd.DataFrame) -> Tuple[Optional[str], str]:
    """Enhanced MACD strategy with volume and trend confirmation"""
    try:
        # Calculate MACD if not already present
        if 'MACD' not in data.columns or 'MACD_signal' not in data.columns:
            data = calculate_macd(data)

        # Add volume confirmation
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        volume_trend = data['volume'] > data['volume_sma']

        # Add trend confirmation
        data['sma_50'] = data['close'].rolling(window=50).mean()
        uptrend = data['close'] > data['sma_50']

        latest = data.iloc[-1]
        previous = data.iloc[-2]

        # Calculate histogram and its direction
        data['MACD_hist'] = data['MACD'] - data['MACD_signal']
        data['MACD_hist_slope'] = data['MACD_hist'] - data['MACD_hist'].shift(1)
        histogram_increasing = latest['MACD_hist_slope'] > 0

        # Strong bullish trend
        if (latest["MACD"] > latest["MACD_signal"] and
                previous["MACD"] <= previous["MACD_signal"]):

            # Check confirmations
            confirmations = []
            if volume_trend.iloc[-1]:
                confirmations.append("increasing volume")
            if uptrend.iloc[-1]:
                confirmations.append("price above SMA50")

            if confirmations:
                return "LONG", f"MACD Bullish Crossover with {', '.join(confirmations)}"
            else:
                return "LONG", "MACD Bullish Crossover"

        # Strong bearish trend
        elif (latest["MACD"] < latest["MACD_signal"] and
              previous["MACD"] >= previous["MACD_signal"]):

            # Check confirmations
            confirmations = []
            if volume_trend.iloc[-1]:
                confirmations.append("increasing volume")
            if not uptrend.iloc[-1]:
                confirmations.append("price below SMA50")

            if confirmations:
                return "SHORT", f"MACD Bearish Crossover with {', '.join(confirmations)}"
            else:
                return "SHORT", "MACD Bearish Crossover"

        # Additional signal: histogram reversal
        elif abs(latest['MACD_hist']) > abs(data['MACD_hist'].mean()) * 1.5:  # Significant histogram
            if latest['MACD'] > 0 and histogram_increasing and uptrend.iloc[-1]:
                return "LONG", "MACD Histogram Bullish Reversal"
            elif latest['MACD'] < 0 and not histogram_increasing and not uptrend.iloc[-1]:
                return "SHORT", "MACD Histogram Bearish Reversal"

        return None, "No MACD signal"

    except Exception as e:
        print(f"Error in MACD trend strategy: {e}")
        return None, f"Error: {str(e)}"


def rsi_strategy(data: pd.DataFrame) -> Tuple[Optional[str], str]:
    """Enhanced RSI strategy with trend and divergence confirmation"""
    try:
        # Get parameters
        period = STRATEGY_PARAMS.get("RSI_OVERBOUGHT_OVERSOLD", {}).get("period", 14)
        overbought = STRATEGY_PARAMS.get("RSI_OVERBOUGHT_OVERSOLD", {}).get("overbought", 70)
        oversold = STRATEGY_PARAMS.get("RSI_OVERBOUGHT_OVERSOLD", {}).get("oversold", 30)

        # Calculate RSI if not already present
        if 'RSI' not in data.columns:
            data = calculate_rsi(data, period)

        # Add trend confirmation
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        trend_up = data['sma_20'] > data['sma_50']

        latest = data.iloc[-1]
        previous_5 = data.iloc[-5:]

        # Check for RSI divergence
        price_making_lower_low = latest['close'] < previous_5['close'].min()
        price_making_higher_high = latest['close'] > previous_5['close'].max()
        rsi_making_higher_low = latest['RSI'] > previous_5['RSI'].min()
        rsi_making_lower_high = latest['RSI'] < previous_5['RSI'].max()

        bullish_divergence = price_making_lower_low and rsi_making_higher_low
        bearish_divergence = price_making_higher_high and rsi_making_lower_high

        # Enhanced logic with divergences and trend
        if latest["RSI"] < oversold:
            if bullish_divergence:
                return "LONG", "RSI Oversold with Bullish Divergence"
            elif trend_up.iloc[-1]:
                return "LONG", "RSI Oversold in Uptrend"
            else:
                return "LONG", "RSI Oversold"

        elif latest["RSI"] > overbought:
            if bearish_divergence:
                return "SHORT", "RSI Overbought with Bearish Divergence"
            elif not trend_up.iloc[-1]:
                return "SHORT", "RSI Overbought in Downtrend"
            else:
                return "SHORT", "RSI Overbought"

        # RSI momentum signals
        elif latest["RSI"] < 40 and latest["RSI"] > latest["RSI"].shift(1) and trend_up.iloc[-1]:
            return "LONG", "RSI Momentum Shift Bullish"

        elif latest["RSI"] > 60 and latest["RSI"] < latest["RSI"].shift(1) and not trend_up.iloc[-1]:
            return "SHORT", "RSI Momentum Shift Bearish"

        return None, "No RSI signal"

    except Exception as e:
        print(f"Error in RSI strategy: {e}")
        return None, f"Error: {str(e)}"


def bollinger_strategy(data: pd.DataFrame) -> Tuple[Optional[str], str]:
    """Bollinger Bands strategy with additional confirmations"""
    try:
        # Calculate Bollinger Bands if not already present
        if 'BB_upper' not in data.columns or 'BB_lower' not in data.columns:
            data = calculate_bollinger_bands(data)

        # Calculate Bollinger Band squeeze
        data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
        data['BB_width_avg'] = data['BB_width'].rolling(window=20).mean()
        is_squeeze = data['BB_width'].iloc[-1] < data['BB_width_avg'].iloc[-1] * 0.8

        # Add RSI for confirmation
        if 'RSI' not in data.columns:
            data = calculate_rsi(data)

        latest = data.iloc[-1]
        previous = data.iloc[-2]

        # Price touching/crossing bands
        price_crossed_lower = previous['close'] >= previous['BB_lower'] and latest['close'] < latest['BB_lower']
        price_crossed_upper = previous['close'] <= previous['BB_upper'] and latest['close'] > latest['BB_upper']

        # Price reversing from bands
        price_bounced_lower = previous['close'] <= previous['BB_lower'] and latest['close'] > latest['BB_lower']
        price_bounced_upper = previous['close'] >= previous['BB_upper'] and latest['close'] < latest['BB_upper']

        # Bollinger Band breakout after squeeze
        if is_squeeze and price_crossed_upper and latest['RSI'] > 50:
            return "LONG", "Bollinger Band Upside Breakout after Squeeze"

        elif is_squeeze and price_crossed_lower and latest['RSI'] < 50:
            return "SHORT", "Bollinger Band Downside Breakout after Squeeze"

        # Bollinger Band mean reversion
        elif price_bounced_lower and latest['RSI'] < 40:
            return "LONG", "Bollinger Band Lower Bounce with RSI Confirmation"

        elif price_bounced_upper and latest['RSI'] > 60:
            return "SHORT", "Bollinger Band Upper Bounce with RSI Confirmation"

        # Price outside bands with extreme RSI
        elif latest['close'] < latest['BB_lower'] and latest['close'] < latest['close'].shift(3) and latest['RSI'] < 30:
            return "LONG", "Oversold Outside Lower Bollinger Band"

        elif latest['close'] > latest['BB_upper'] and latest['close'] > latest['close'].shift(3) and latest['RSI'] > 70:
            return "SHORT", "Overbought Outside Upper Bollinger Band"

        return None, "No Bollinger Band signal"

    except Exception as e:
        print(f"Error in Bollinger Band strategy: {e}")
        return None, f"Error: {str(e)}"


def supertrend_strategy(data: pd.DataFrame) -> Tuple[Optional[str], str]:
    """SuperTrend strategy for trend following"""
    try:
        # Calculate SuperTrend if not already present
        if 'SuperTrend' not in data.columns:
            data = calculate_supertrend(data)

        # Get trend direction
        data['ST_UpTrend'] = data['close'] > data['SuperTrend']

        # Check for trend change
        trend_changed_bullish = data['ST_UpTrend'].iloc[-1] and not data['ST_UpTrend'].iloc[-2]
        trend_changed_bearish = not data['ST_UpTrend'].iloc[-1] and data['ST_UpTrend'].iloc[-2]

        # Check trend strength
        in_uptrend_3_bars = data['ST_UpTrend'].iloc[-3:].all()
        in_downtrend_3_bars = (~data['ST_UpTrend']).iloc[-3:].all()

        # Signals
        if trend_changed_bullish:
            return "LONG", "SuperTrend Bullish Signal"

        elif trend_changed_bearish:
            return "SHORT", "SuperTrend Bearish Signal"

        # Additional signals: Price pulling back to SuperTrend in established trend
        elif in_uptrend_3_bars and (data['low'].iloc[-1] - data['SuperTrend'].iloc[-1]) / data['SuperTrend'].iloc[
            -1] < 0.003:
            return "LONG", "SuperTrend Bullish Pullback"

        elif in_downtrend_3_bars and (data['SuperTrend'].iloc[-1] - data['high'].iloc[-1]) / data['high'].iloc[
            -1] < 0.003:
            return "SHORT", "SuperTrend Bearish Pullback"

        return None, "No SuperTrend signal"

    except Exception as e:
        print(f"Error in SuperTrend strategy: {e}")
        return None, f"Error: {str(e)}"


def multi_timeframe_strategy(data_main: pd.DataFrame,
                             data_short: pd.DataFrame,
                             data_long: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Multi-timeframe confirmations strategy
    Uses short timeframe for momentum, main for signals, long for trend direction
    """
    try:
        # Calculate moving averages on all timeframes if not present
        for df, label in [(data_main, "main"), (data_short, "short"), (data_long, "long")]:
            if 'EMA_50' not in df.columns:
                df = calculate_ema(df, 50)
            if 'EMA_200' not in df.columns:
                df = calculate_ema(df, 200)
            if 'RSI' not in df.columns:
                df = calculate_rsi(df)

        # Check trend alignment
        short_trend = data_short['close'].iloc[-1] > data_short['EMA_50'].iloc[-1]
        main_trend = data_main['close'].iloc[-1] > data_main['EMA_50'].iloc[-1]
        long_trend = data_long['close'].iloc[-1] > data_long['EMA_50'].iloc[-1]

        # Calculate momentum
        short_momentum = data_short['RSI'].iloc[-1] > 50
        main_momentum = data_main['RSI'].iloc[-1] > 50

        # Check for aligned bullish signals
        if long_trend and main_trend and short_trend and short_momentum and main_momentum:
            return "LONG", "Multi-Timeframe Bullish Alignment"

        # Check for aligned bearish signals
        elif not long_trend and not main_trend and not short_trend and not short_momentum and not main_momentum:
            return "SHORT", "Multi-Timeframe Bearish Alignment"

        # Check for pending reversal (long timeframe trend vs short timeframe momentum)
        elif long_trend and not main_trend and short_momentum:
            return "LONG", "Multi-Timeframe Bullish Reversal"

        elif not long_trend and main_trend and not short_momentum:
            return "SHORT", "Multi-Timeframe Bearish Reversal"

        return None, "No Multi-Timeframe signal"

    except Exception as e:
        print(f"Error in Multi-Timeframe strategy: {e}")
        return None, f"Error: {str(e)}"