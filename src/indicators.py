"""
Enhanced Technical Indicators Module
- Comprehensive set of technical indicators
- Optimized for performance
- Support for custom parameters
- Multi-timeframe analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import ta
from scipy import stats


def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate multiple moving averages on price data
    Includes: SMA, EMA, WMA, HMA
    """
    if df is None or len(df) < 50:
        return df

    try:
        # Simple Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)

        # Exponential Moving Averages
        df['EMA_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['EMA_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['EMA_200'] = ta.trend.ema_indicator(df['close'], window=200)

        # Weighted Moving Average
        def wma(data, window):
            weights = np.arange(1, window + 1)
            return data.rolling(window).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)

        df['WMA_20'] = wma(df['close'], 20)

        # Hull Moving Average
        def hma(data, window):
            return wma(2 * wma(data, window // 2) - wma(data, window), int(np.sqrt(window)))

        df['HMA_20'] = hma(df['close'], 20)

        # Moving Average Convergence Divergence
        df = calculate_macd(df)

        return df

    except Exception as e:
        print(f"Error calculating moving averages: {e}")
        return df


def calculate_ema(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Exponential Moving Average"""
    if df is None or len(df) < period:
        return df

    try:
        df[f"EMA_{period}"] = ta.trend.ema_indicator(df["close"], window=period)
        return df

    except Exception as e:
        print(f"Error calculating EMA: {e}")
        return df


def calculate_macd(df: pd.DataFrame,
                   fast_period: int = 12,
                   slow_period: int = 26,
                   signal_period: int = 9) -> pd.DataFrame:
    """Calculate MACD and Signal Line"""
    if df is None or len(df) < slow_period + signal_period:
        return df

    try:
        # Calculate MACD Line
        fast_ema = ta.trend.ema_indicator(df["close"], window=fast_period)
        slow_ema = ta.trend.ema_indicator(df["close"], window=slow_period)
        df["MACD"] = fast_ema - slow_ema

        # Calculate Signal Line
        df["MACD_signal"] = ta.trend.ema_indicator(df["MACD"], window=signal_period)

        # Calculate Histogram
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

        return df

    except Exception as e:
        print(f"Error calculating MACD: {e}")
        return df


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Relative Strength Index"""
    if df is None or len(df) < period:
        return df

    try:
        df["RSI"] = ta.momentum.rsi(df["close"], window=period)

        # Calculate overbought/oversold zones
        df["RSI_overbought"] = df["RSI"] > 70
        df["RSI_oversold"] = df["RSI"] < 30

        # Calculate RSI direction
        df["RSI_up"] = df["RSI"] > df["RSI"].shift(1)

        return df

    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return df


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator"""
    if df is None or len(df) < k_period + d_period:
        return df

    try:
        df["Stoch_K"] = ta.momentum.stoch(df["high"], df["low"], df["close"], window=k_period, smooth_window=1)
        df["Stoch_D"] = df["Stoch_K"].rolling(window=d_period).mean()

        # Calculate overbought/oversold zones
        df["Stoch_overbought"] = df["Stoch_K"] > 80
        df["Stoch_oversold"] = df["Stoch_K"] < 20

        # Calculate crossovers
        df["Stoch_bullish_cross"] = (df["Stoch_K"] > df["Stoch_D"]) & (df["Stoch_K"].shift(1) <= df["Stoch_D"].shift(1))
        df["Stoch_bearish_cross"] = (df["Stoch_K"] < df["Stoch_D"]) & (df["Stoch_K"].shift(1) >= df["Stoch_D"].shift(1))

        return df

    except Exception as e:
        print(f"Error calculating Stochastic: {e}")
        return df


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands"""
    if df is None or len(df) < period:
        return df

    try:
        bb = ta.volatility.BollingerBands(df["close"], window=period, window_dev=std_dev)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_middle"] = bb.bollinger_mavg()
        df["BB_lower"] = bb.bollinger_lband()

        # Calculate bands width (volatility)
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]

        # Calculate position relative to bands
        df["BB_pct_b"] = (df["close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

        return df

    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")
        return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Average True Range for volatility measurement"""
    if df is None or len(df) < period:
        return df

    try:
        # Calculate True Range
        df["TR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=period,
                                                    fillna=True) * period

        # Calculate ATR
        df["ATR"] = df["TR"] / period

        # Calculate ATR percent (relative to price)
        df["ATR_pct"] = df["ATR"] / df["close"] * 100

        return df

    except Exception as e:
        print(f"Error calculating ATR: {e}")
        return df


def calculate_supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Calculate SuperTrend indicator

    SuperTrend = 
    - If current price > previous SuperTrend and current price > basic upper band, then basic upper band
    - If current price < previous SuperTrend and current price < basic lower band, then basic lower band
    - Otherwise, previous SuperTrend
    """
    if df is None or len(df) < atr_period:
        return df

    try:
        # Calculate ATR if not present
        if "ATR" not in df.columns:
            df = calculate_atr(df, atr_period)

        # Basic Bands
        df["basic_upper_band"] = ((df["high"] + df["low"]) / 2) + (multiplier * df["ATR"])
        df["basic_lower_band"] = ((df["high"] + df["low"]) / 2) - (multiplier * df["ATR"])

        # Initialize SuperTrend
        df["SuperTrend"] = 0.0
        df["ST_direction"] = 0  # 1 for uptrend, -1 for downtrend

        # Calculate SuperTrend iteratively
        for i in range(1, len(df)):
            # Direction
            if df["close"].iloc[i - 1] <= df["SuperTrend"].iloc[i - 1]:
                df.loc[df.index[i], "ST_direction"] = -1  # Downtrend
            else:
                df.loc[df.index[i], "ST_direction"] = 1  # Uptrend

            # SuperTrend value
            if (df["ST_direction"].iloc[i] == 1 and
                    df["basic_lower_band"].iloc[i] < df["SuperTrend"].iloc[i - 1]):
                df.loc[df.index[i], "SuperTrend"] = df["basic_lower_band"].iloc[i]
            elif (df["ST_direction"].iloc[i] == 1 and
                  df["basic_lower_band"].iloc[i] >= df["SuperTrend"].iloc[i - 1]):
                df.loc[df.index[i], "SuperTrend"] = df["SuperTrend"].iloc[i - 1]
            elif (df["ST_direction"].iloc[i] == -1 and
                  df["basic_upper_band"].iloc[i] > df["SuperTrend"].iloc[i - 1]):
                df.loc[df.index[i], "SuperTrend"] = df["basic_upper_band"].iloc[i]
            elif (df["ST_direction"].iloc[i] == -1 and
                  df["basic_upper_band"].iloc[i] <= df["SuperTrend"].iloc[i - 1]):
                df.loc[df.index[i], "SuperTrend"] = df["SuperTrend"].iloc[i - 1]

        return df

    except Exception as e:
        print(f"Error calculating SuperTrend: {e}")
        return df


def calculate_ichimoku(df: pd.DataFrame,
                       tenkan_period: int = 9,
                       kijun_period: int = 26,
                       senkou_span_b_period: int = 52) -> pd.DataFrame:
    """Calculate Ichimoku Cloud components"""
    if df is None or len(df) < senkou_span_b_period:
        return df

    try:
        # Tenkan-sen (Conversion Line)
        tenkan_high = df["high"].rolling(window=tenkan_period).max()
        tenkan_low = df["low"].rolling(window=tenkan_period).min()
        df["ichimoku_tenkan"] = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = df["high"].rolling(window=kijun_period).max()
        kijun_low = df["low"].rolling(window=kijun_period).min()
        df["ichimoku_kijun"] = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A)
        df["ichimoku_senkou_a"] = ((df["ichimoku_tenkan"] + df["ichimoku_kijun"]) / 2).shift(kijun_period)

        # Senkou Span B (Leading Span B)
        senkou_high = df["high"].rolling(window=senkou_span_b_period).max()
        senkou_low = df["low"].rolling(window=senkou_span_b_period).min()
        df["ichimoku_senkou_b"] = ((senkou_high + senkou_low) / 2).shift(kijun_period)

        # Chikou Span (Lagging Span)
        df["ichimoku_chikou"] = df["close"].shift(-kijun_period)

        # Calculate cloud direction and thickness
        df["ichimoku_cloud_green"] = df["ichimoku_senkou_a"] > df["ichimoku_senkou_b"]
        df["ichimoku_cloud_thickness"] = abs(df["ichimoku_senkou_a"] - df["ichimoku_senkou_b"])

        # Calculate TK Cross
        df["ichimoku_tk_cross_bullish"] = (df["ichimoku_tenkan"] > df["ichimoku_kijun"]) & (
                    df["ichimoku_tenkan"].shift(1) <= df["ichimoku_kijun"].shift(1))
        df["ichimoku_tk_cross_bearish"] = (df["ichimoku_tenkan"] < df["ichimoku_kijun"]) & (
                    df["ichimoku_tenkan"].shift(1) >= df["ichimoku_kijun"].shift(1))

        return df

    except Exception as e:
        print(f"Error calculating Ichimoku: {e}")
        return df


def calculate_volume_profile(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Calculate Volume Profile to identify high volume price levels"""
    if df is None or len(df) < 50:
        return df

    try:
        # Calculate price range for the period
        price_min = df["low"].min()
        price_max = df["high"].max()
        price_range = price_max - price_min

        # Create price bins
        bin_size = price_range / bins
        bin_edges = [price_min + i * bin_size for i in range(bins + 1)]
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(bins)]

        # Calculate volume in each price bin
        volume_profile = np.zeros(bins)

        for i in range(len(df)):
            # Use typical price
            typical_price = (df["high"].iloc[i] + df["low"].iloc[i] + df["close"].iloc[i]) / 3

            # Find bin index
            if typical_price <= price_min:
                bin_idx = 0
            elif typical_price >= price_max:
                bin_idx = bins - 1
            else:
                bin_idx = int((typical_price - price_min) / bin_size)

            # Add volume to the bin
            volume_profile[bin_idx] += df["volume"].iloc[i]

        # Find Point of Control (POC) - price level with highest volume
        poc_idx = np.argmax(volume_profile)
        poc_price = bin_centers[poc_idx]

        # Find Value Area (70% of volume)
        total_volume = np.sum(volume_profile)
        value_area_threshold = total_volume * 0.7

        # Sort bins by volume
        sorted_indices = np.argsort(volume_profile)[::-1]
        cumulative_volume = 0
        value_area_bins = []

        for idx in sorted_indices:
            cumulative_volume += volume_profile[idx]
            value_area_bins.append(idx)
            if cumulative_volume >= value_area_threshold:
                break

        # Find Value Area High (VAH) and Value Area Low (VAL)
        if value_area_bins:
            val_price = bin_centers[min(value_area_bins)]
            vah_price = bin_centers[max(value_area_bins)]
        else:
            val_price = price_min
            vah_price = price_max

        # Add results to dataframe
        df["VP_POC"] = poc_price
        df["VP_VAL"] = val_price
        df["VP_VAH"] = vah_price

        # Check if current price is at key levels
        current_price = df["close"].iloc[-1]
        df["VP_at_POC"] = abs(current_price - poc_price) / poc_price < 0.005  # Within 0.5% of POC
        df["VP_at_VAL"] = abs(current_price - val_price) / val_price < 0.005
        df["VP_at_VAH"] = abs(current_price - vah_price) / vah_price < 0.005

        return df

    except Exception as e:
        print(f"Error calculating Volume Profile: {e}")
        return df


def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate multiple momentum indicators"""
    if df is None or len(df) < 50:
        return df

    try:
        # Calculate RSI if not present
        if "RSI" not in df.columns:
            df = calculate_rsi(df)

        # Stochastic Oscillator
        df = calculate_stochastic(df)

        # Commodity Channel Index (CCI)
        df["CCI"] = ta.trend.cci(df["high"], df["low"], df["close"], window=20)

        # Money Flow Index (MFI)
        df["MFI"] = ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"], window=14)

        # Rate of Change (ROC)
        df["ROC"] = ta.momentum.roc(df["close"], window=10)

        # Williams %R
        df["Williams_R"] = ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=14)

        # Average Directional Index (ADX)
        df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
        df["ADX_strong_trend"] = df["ADX"] > 25

        # Calculate momentum divergence
        # Price making higher high but momentum not confirming
        df["close_higher_high"] = (df["close"] > df["close"].shift(1)) & (df["close"].shift(1) > df["close"].shift(2))
        df["rsi_lower_high"] = (df["RSI"] < df["RSI"].shift(1)) & (df["RSI"].shift(1) > df["RSI"].shift(2))
        df["bearish_divergence"] = df["close_higher_high"] & df["rsi_lower_high"]

        # Price making lower low but momentum not confirming
        df["close_lower_low"] = (df["close"] < df["close"].shift(1)) & (df["close"].shift(1) < df["close"].shift(2))
        df["rsi_higher_low"] = (df["RSI"] > df["RSI"].shift(1)) & (df["RSI"].shift(1) < df["RSI"].shift(2))
        df["bullish_divergence"] = df["close_lower_low"] & df["rsi_higher_low"]

        return df

    except Exception as e:
        print(f"Error calculating momentum indicators: {e}")
        return df


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Volume Weighted Average Price"""
    if df is None or len(df) < 1:
        return df

    try:
        # Calculate typical price
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

        # Calculate VWAP
        df["VWAP"] = (df["typical_price"] * df["volume"]).cumsum() / df["volume"].cumsum()

        # Calculate distance from VWAP
        df["VWAP_distance"] = (df["close"] - df["VWAP"]) / df["VWAP"] * 100  # As percentage

        return df

    except Exception as e:
        print(f"Error calculating VWAP: {e}")
        return df


def calculate_fibonacci_levels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Calculate Fibonacci retracement levels based on recent swing high/low"""
    if df is None or len(df) < period:
        return df

    try:
        # Find recent swing high and low
        window = df.iloc[-period:]
        swing_high = window["high"].max()
        swing_low = window["low"].min()
        price_range = swing_high - swing_low

        # Calculate Fibonacci levels
        fib_levels = {
            "0.0": swing_low,
            "0.236": swing_low + 0.236 * price_range,
            "0.382": swing_low + 0.382 * price_range,
            "0.5": swing_low + 0.5 * price_range,
            "0.618": swing_low + 0.618 * price_range,
            "0.786": swing_low + 0.786 * price_range,
            "1.0": swing_high
        }

        # Add Fibonacci extension levels
        fib_levels["1.272"] = swing_low + 1.272 * price_range
        fib_levels["1.618"] = swing_low + 1.618 * price_range

        # Add to DataFrame
        for level, price in fib_levels.items():
            df[f"Fib_{level}"] = price

        # Check if price is at a Fibonacci level
        current_price = df["close"].iloc[-1]
        for level, price in fib_levels.items():
            # Check if within 0.5% of level
            df[f"At_Fib_{level}"] = abs(current_price - price) / price < 0.005

        return df

    except Exception as e:
        print(f"Error calculating Fibonacci levels: {e}")
        return df


def calculate_support_resistance(df: pd.DataFrame, period: int = 50, min_touches: int = 2,
                                 deviation_pct: float = 0.005) -> Dict:
    """Identify support and resistance levels based on price history"""
    if df is None or len(df) < period:
        return {"support": [], "resistance": []}

    try:
        # Use a subset of recent data
        window = df.iloc[-period:]

        # Find potential levels from highs and lows
        highs = window["high"].tolist()
        lows = window["low"].tolist()

        # Cluster nearby levels
        def cluster_levels(prices, deviation):
            clusters = []
            for price in sorted(prices):
                # Check if price fits in an existing cluster
                found_cluster = False
                for i, cluster in enumerate(clusters):
                    if abs(price - cluster["mean"]) / cluster["mean"] <= deviation:
                        cluster["prices"].append(price)
                        cluster["mean"] = sum(cluster["prices"]) / len(cluster["prices"])
                        cluster["count"] += 1
                        found_cluster = True
                        break

                # If not, create a new cluster
                if not found_cluster:
                    clusters.append({
                        "prices": [price],
                        "mean": price,
                        "count": 1
                    })

            return [c for c in clusters if c["count"] >= min_touches]

        # Cluster highs and lows
        resistance_clusters = cluster_levels(highs, deviation_pct)
        support_clusters = cluster_levels(lows, deviation_pct)

        # Extract mean prices
        resistance_levels = sorted([cluster["mean"] for cluster in resistance_clusters])
        support_levels = sorted([cluster["mean"] for cluster in support_clusters])

        # Filter out levels that are too close
        def filter_too_close(levels, min_distance):
            if not levels:
                return []

            filtered = [levels[0]]
            for level in levels[1:]:
                if abs(level - filtered[-1]) / filtered[-1] > min_distance:
                    filtered.append(level)

            return filtered

        resistance_levels = filter_too_close(resistance_levels, deviation_pct * 2)
        support_levels = filter_too_close(support_levels, deviation_pct * 2)

        # Return both lists
        return {
            "support": support_levels,
            "resistance": resistance_levels
        }

    except Exception as e:
        print(f"Error calculating support and resistance: {e}")
        return {"support": [], "resistance": []}


def calculate_multi_timeframe(symbols: List[str], timeframes: List[str], client) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Calculate indicators across multiple timeframes for a list of symbols"""
    result = {}

    for symbol in symbols:
        result[symbol] = {}

        for timeframe in timeframes:
            try:
                # Fetch historical data
                klines = client.futures_klines(symbol=symbol, interval=timeframe, limit=500)
                df = pd.DataFrame(klines, columns=[
                    "time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore"
                ])

                # Convert to numeric
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col])

                # Convert timestamp
                df["time"] = pd.to_datetime(df["time"], unit="ms")

                # Calculate indicators
                df = calculate_moving_averages(df)
                df = calculate_momentum_indicators(df)
                df = calculate_bollinger_bands(df)
                df = calculate_vwap(df)

                # Store in result
                result[symbol][timeframe] = df

            except Exception as e:
                print(f"Error calculating indicators for {symbol} on {timeframe}: {e}")
                result[symbol][timeframe] = None

    return result


def detect_chart_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """Detect common chart patterns"""
    if df is None or len(df) < 50:
        return {}

    try:
        patterns = {}

        # Head and Shoulders pattern
        # (Requires 5 points: left shoulder peak, head peak, right shoulder peak, and two troughs)
        if len(df) >= 25:
            window = df.iloc[-25:].copy()

            # Find peaks and troughs
            window["peak"] = (window["high"] > window["high"].shift(1)) & (window["high"] > window["high"].shift(-1))
            window["trough"] = (window["low"] < window["low"].shift(1)) & (window["low"] < window["low"].shift(-1))

            peaks = window[window["peak"]]["high"].tolist()
            troughs = window[window["trough"]]["low"].tolist()

            if len(peaks) >= 3 and len(troughs) >= 2:
                # Check if middle peak is higher than outer peaks
                peaks_sorted = sorted(peaks, reverse=True)
                if peaks.index(peaks_sorted[0]) > peaks.index(peaks_sorted[1]) and peaks.index(
                        peaks_sorted[0]) < peaks.index(peaks_sorted[2]):
                    patterns["head_and_shoulders"] = True
                else:
                    patterns["head_and_shoulders"] = False
            else:
                patterns["head_and_shoulders"] = False

        # Double Top pattern
        if len(df) >= 20:
            window = df.iloc[-20:].copy()

            # Find peaks
            window["peak"] = (window["high"] > window["high"].shift(1)) & (window["high"] > window["high"].shift(-1))
            peaks = window[window["peak"]]["high"].tolist()

            if len(peaks) >= 2:
                # Check if two highest peaks are within 2% of each other
                peaks_sorted = sorted(peaks, reverse=True)
                if abs(peaks_sorted[0] - peaks_sorted[1]) / peaks_sorted[1] < 0.02:
                    patterns["double_top"] = True
                else:
                    patterns["double_top"] = False
            else:
                patterns["double_top"] = False

        # Double Bottom pattern
        if len(df) >= 20:
            window = df.iloc[-20:].copy()

            # Find troughs
            window["trough"] = (window["low"] < window["low"].shift(1)) & (window["low"] < window["low"].shift(-1))
            troughs = window[window["trough"]]["low"].tolist()

            if len(troughs) >= 2:
                # Check if two lowest troughs are within 2% of each other
                troughs_sorted = sorted(troughs)
                if abs(troughs_sorted[0] - troughs_sorted[1]) / troughs_sorted[1] < 0.02:
                    patterns["double_bottom"] = True
                else:
                    patterns["double_bottom"] = False
            else:
                patterns["double_bottom"] = False

        # Bull Flag pattern
        if len(df) >= 20:
            # Check for strong uptrend followed by consolidation
            uptrend = df["close"].iloc[-20:-10].pct_change().sum() > 0.05  # 5% uptrend
            consolidation = abs(df["close"].iloc[-10:].pct_change().sum()) < 0.02  # Less than 2% change

            if uptrend and consolidation:
                patterns["bull_flag"] = True
            else:
                patterns["bull_flag"] = False

        # Bear Flag pattern
        if len(df) >= 20:
            # Check for strong downtrend followed by consolidation
            downtrend = df["close"].iloc[-20:-10].pct_change().sum() < -0.05  # -5% downtrend
            consolidation = abs(df["close"].iloc[-10:].pct_change().sum()) < 0.02  # Less than 2% change

            if downtrend and consolidation:
                patterns["bear_flag"] = True
            else:
                patterns["bear_flag"] = False

        return patterns

    except Exception as e:
        print(f"Error detecting chart patterns: {e}")
        return {}