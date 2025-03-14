"""
Enhanced Binance Futures Trading Bot
- Multi-strategy with confirmation
- Advanced risk management
- Real-time data via WebSockets
- Position and drawdown management
- Market regime detection
- Performance analytics
"""

import time
import pandas as pd
import numpy as np
import json
import threading
import asyncio
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL
from binance.exceptions import BinanceAPIException
import traceback

# Import custom modules
from src.config import (
    BINANCE_API_KEY, BINANCE_SECRET_KEY, FUTURES_SYMBOLS,
    MAX_DRAWDOWN, TRADING_STRATEGIES, DEFAULT_LEVERAGE,
    USE_WEBSOCKETS, TRADE_TIMEOUT_HOURS, BACKTEST_MODE
)
from src.trade_manager import (
    place_order, set_leverage, calculate_position_size,
    update_trailing_stop, close_position, adjust_risk_based_on_market
)
from src.strategies import (
    should_enter_trade, should_exit_trade, analyze_market_conditions,
    detect_market_regime, get_strategy_combination_score
)
from src.analytics import (
    update_trade_performance, log_daily_statistics,
    notify_important_event, calculate_sharpe_ratio
)
from src.logger import log_trade, log_trade_exit, log_error, log_info
from src.websocket_manager import BinanceWebSocketManager

# Initialize Binance Futures Client
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# Initialize WebSocket Manager if enabled
ws_manager = None
if USE_WEBSOCKETS:
    ws_manager = BinanceWebSocketManager(FUTURES_SYMBOLS, BINANCE_API_KEY, BINANCE_SECRET_KEY)

# Set leverage for all trading pairs
for symbol in FUTURES_SYMBOLS:
    try:
        set_leverage(symbol, DEFAULT_LEVERAGE)
        log_info(f"Leverage set to {DEFAULT_LEVERAGE}x for {symbol}")
    except Exception as e:
        log_error(f"Error setting leverage for {symbol}: {e}")

# Global data cache
market_data_cache = {}
open_positions = {}
trade_history = []
last_health_check = datetime.now()


def get_balance() -> float:
    """Get USDT balance from futures account"""
    try:
        balance_info = client.futures_account_balance()
        for asset in balance_info:
            if asset["asset"] == "USDT":
                return float(asset["balance"])
        return 0.0
    except Exception as e:
        log_error(f"Error getting balance: {e}")
        return 0.0


def get_historical_data(symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """Fetch historical candlestick data with error handling and processing"""
    try:
        log_info(f"Fetching market data for {symbol} ({interval})")
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)

        # Create DataFrame with proper column names
        df = pd.DataFrame(klines, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        # Convert columns to numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['date'] = df['time'].dt.date

        # Calculate additional base metrics
        df['range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['is_bullish'] = df['close'] > df['open']

        # Cache the processed data
        market_data_cache[f"{symbol}_{interval}"] = df

        return df
    except Exception as e:
        log_error(f"Error fetching data for {symbol}: {e}")
        # If we have cached data, return that instead of None
        if f"{symbol}_{interval}" in market_data_cache:
            log_info(f"Using cached data for {symbol}")
            return market_data_cache[f"{symbol}_{interval}"]
        return None


def check_drawdown(initial_balance: float, highest_balance: float) -> Tuple[bool, float, float]:
    """
    Check drawdown compared to initial and peak balances
    Returns: (continue_trading, current_drawdown, risk_multiplier)
    """
    try:
        current_balance = get_balance()
        if current_balance <= 0:
            log_error("Zero or negative balance detected")
            return False, 1.0, 0.0

        # Calculate drawdowns
        initial_drawdown = (initial_balance - current_balance) / initial_balance
        peak_drawdown = (highest_balance - current_balance) / highest_balance

        # Use the worse of the two drawdowns
        current_drawdown = max(initial_drawdown, peak_drawdown)

        # Dynamic risk adjustment based on drawdown
        risk_multiplier = 1.0
        if current_drawdown > abs(MAX_DRAWDOWN * 0.5):
            # Linear reduction from 100% to 25% as drawdown approaches max
            risk_reduction_factor = (current_drawdown - MAX_DRAWDOWN * 0.5) / (MAX_DRAWDOWN * 0.5)
            risk_multiplier = max(0.25, 1.0 - (risk_reduction_factor * 0.75))
            log_info(f"Drawdown at {current_drawdown:.2%} - Reducing risk to {risk_multiplier:.2%}")

        # If we hit maximum drawdown, stop trading
        if current_drawdown > abs(MAX_DRAWDOWN):
            log_error(f"Maximum drawdown reached: {current_drawdown:.2%}")
            return False, current_drawdown, risk_multiplier

        return True, current_drawdown, risk_multiplier

    except Exception as e:
        log_error(f"Error checking drawdown: {e}")
        return False, 1.0, 0.0


def get_position_info(symbol: str) -> Optional[Dict]:
    """Get current position information with enhanced error handling"""
    try:
        # If we're using WebSockets, try to get the position from the cache first
        if ws_manager and symbol in ws_manager.position_cache:
            return ws_manager.position_cache[symbol]

        # Otherwise get from API
        positions = client.futures_position_information()
        for position in positions:
            if position['symbol'] == symbol:
                pos_data = {
                    'symbol': symbol,
                    'amount': float(position['positionAmt']),
                    'entry_price': float(position['entryPrice']),
                    'mark_price': float(position['markPrice']),
                    'unrealized_pnl': float(position['unRealizedProfit']),
                    'liquidation_price': float(position['liquidationPrice']) if float(
                        position['liquidationPrice']) > 0 else None,
                    'leverage': float(position['leverage']),
                    'margin_type': position['marginType'],
                    'position_side': position['positionSide'],
                    'update_time': datetime.now()
                }

                # Calculate additional metrics
                if pos_data['amount'] != 0:
                    pos_data['side'] = "LONG" if pos_data['amount'] > 0 else "SHORT"
                    pos_data['pnl_pct'] = pos_data['unrealized_pnl'] / (
                                abs(pos_data['amount']) * pos_data['entry_price']) * 100
                    pos_data['value'] = abs(pos_data['amount'] * pos_data['mark_price'])

                    # Check if this position is in our tracked open positions
                    if symbol in open_positions:
                        pos_data['stop_loss'] = open_positions[symbol].get('stop_loss')
                        pos_data['take_profit'] = open_positions[symbol].get('take_profit')
                        pos_data['strategy'] = open_positions[symbol].get('strategy')
                        pos_data['entry_time'] = open_positions[symbol].get('entry_time')
                        pos_data['position_duration'] = datetime.now() - open_positions[symbol].get('entry_time',
                                                                                                    datetime.now())

                return pos_data

        # No position found with non-zero amount
        return None

    except Exception as e:
        log_error(f"Error getting position info for {symbol}: {e}")
        # Return None or cached position if available
        if symbol in open_positions:
            return open_positions[symbol]
        return None


def manage_open_position(symbol: str, position: Dict, market_data: pd.DataFrame) -> bool:
    """
    Manage an existing position - check for exit conditions, update trailing stops
    Returns: True if position was closed, False if still open
    """
    try:
        # Check if position timeout exceeded
        if 'entry_time' in position:
            position_age = datetime.now() - position['entry_time']
            if position_age > timedelta(hours=TRADE_TIMEOUT_HOURS):
                log_info(f"Position timeout exceeded ({TRADE_TIMEOUT_HOURS}h) for {symbol}")
                close_position(symbol, position['side'], "Timeout Exit")
                if symbol in open_positions:
                    del open_positions[symbol]
                return True

        current_price = position['mark_price']

        # Calculate current profit/loss percentage
        if position['side'] == "LONG":
            profit_pct = (current_price - position['entry_price']) / position['entry_price']
        else:  # SHORT trade
            profit_pct = (position['entry_price'] - current_price) / position['entry_price']

        # Get market conditions and regime
        market_conditions = analyze_market_conditions(market_data)
        market_regime = detect_market_regime(market_data)

        # Get exit signal from strategy
        exit_trade, exit_reason = should_exit_trade(
            position['entry_price'],
            current_price,
            position['side'],
            market_conditions,
            market_regime,
            profit_pct,
            position_age=position.get('position_duration')
        )

        # If exit signal or stop conditions met
        if exit_trade:
            log_info(f"Closing {position['side']} trade for {symbol}: {exit_reason}")
            close_position(symbol, position['side'], exit_reason)

            # Track trade results
            trade_result = {
                'symbol': symbol,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'profit_pct': profit_pct * 100,
                'profit_amount': position['unrealized_pnl'],
                'duration': position.get('position_duration', timedelta(0)),
                'exit_reason': exit_reason,
                'exit_time': datetime.now(),
                'strategy': position.get('strategy', 'Unknown')
            }

            # Add to trade history and update analytics
            trade_history.append(trade_result)
            update_trade_performance(trade_result)

            # Remove from tracked positions
            if symbol in open_positions:
                del open_positions[symbol]

            return True

        # If still in position, update trailing stop if enabled
        update_trailing_stop(
            symbol=symbol,
            entry_price=position['entry_price'],
            current_price=current_price,
            trade_type=position['side']
        )

        return False

    except Exception as e:
        log_error(f"Error managing position for {symbol}: {e}")
        return False


def execute_new_trade(symbol: str, trade_signal: str, reason: str,
                      current_price: float, market_data: pd.DataFrame,
                      risk_multiplier: float, score: float) -> bool:
    """
    Execute a new trade based on signal
    Returns: True if trade was executed, False otherwise
    """
    try:
        # Extra safety check to prevent duplicate trades
        position = get_position_info(symbol)
        if position and abs(position['amount']) > 0:
            log_info(f"Position already exists for {symbol}, skipping trade")
            return False

        # Calculate stop loss and take profit levels based on market volatility
        market_conditions = analyze_market_conditions(market_data)
        atr = market_conditions.get('volatility', 0)

        # Dynamic SL/TP based on ATR and market regime
        stop_loss_multiplier = 1.5  # Default 1.5x ATR for stop loss
        take_profit_multiplier = 2.5  # Default 2.5x ATR for take profit

        # Adjust based on market regime
        market_regime = detect_market_regime(market_data)
        if market_regime == 'trending':
            # In trending markets: tighter SL, wider TP
            stop_loss_multiplier = 1.25
            take_profit_multiplier = 3.0
        elif market_regime == 'volatile':
            # In volatile markets: wider SL, tighter TP
            stop_loss_multiplier = 2.0
            take_profit_multiplier = 2.0
        elif market_regime == 'ranging':
            # In ranging markets: moderate settings
            stop_loss_multiplier = 1.5
            take_profit_multiplier = 1.5

        # Calculate actual SL/TP prices
        if trade_signal == "LONG":
            stop_loss = round(current_price * (1 - (atr * stop_loss_multiplier / current_price)), 2)
            take_profit = round(current_price * (1 + (atr * take_profit_multiplier / current_price)), 2)
        else:  # SHORT
            stop_loss = round(current_price * (1 + (atr * stop_loss_multiplier / current_price)), 2)
            take_profit = round(current_price * (1 - (atr * take_profit_multiplier / current_price)), 2)

        # Adjust risk based on strategy score
        signal_risk_multiplier = min(1.0, score / 100) * risk_multiplier

        # Execute trade with appropriate risk
        log_info(f"Executing {trade_signal} for {symbol} with {signal_risk_multiplier:.2f}x risk. Reason: {reason}")
        order = place_order(
            symbol=symbol,
            side=SIDE_BUY if trade_signal == "LONG" else SIDE_SELL,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_multiplier=signal_risk_multiplier
        )

        if order:
            # Track the new position
            entry_data = {
                'symbol': symbol,
                'side': trade_signal,
                'entry_price': current_price,
                'quantity': float(order['origQty']),
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': reason,
                'score': score
            }
            open_positions[symbol] = entry_data

            # Log the trade
            log_trade(
                symbol=symbol,
                trade_type=trade_signal,
                quantity=order['origQty'],
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status="EXECUTED"
            )

            # Notify important trade event
            notify_important_event(f"New {trade_signal} position opened for {symbol} at {current_price}")

            return True

        return False

    except Exception as e:
        log_error(f"Error executing trade for {symbol}: {e}")
        return False


def system_health_check() -> bool:
    """Perform periodic system health check"""
    global last_health_check

    # Only run health check every 5 minutes
    if (datetime.now() - last_health_check).total_seconds() < 300:
        return True

    try:
        # Check API connectivity
        time_response = client.get_server_time()
        if not time_response:
            log_error("Unable to connect to Binance API")
            return False

        # Check account status
        account = client.futures_account()
        if float(account['totalMarginBalance']) <= 0:
            log_error("Account margin balance is zero or negative")
            return False

        # Check WebSocket status if enabled
        if USE_WEBSOCKETS and ws_manager:
            if not ws_manager.is_connected():
                log_error("WebSocket connection lost")
                ws_manager.reconnect()

        # Log successful health check
        last_health_check = datetime.now()
        return True

    except Exception as e:
        log_error(f"Health check failed: {e}")
        return False


async def run_bot_async():
    """Asynchronous bot main loop"""
    print("🚀 Enhanced Binance Futures Trading Bot Started 🚀")

    # Get initial account state
    initial_balance = get_balance()
    if initial_balance <= 0:
        log_error("Invalid initial balance. Cannot start trading.")
        return

    log_info(f"Initial Balance: ${initial_balance:.2f} USDT")

    # Track highest balance for drawdown calculation
    highest_balance = initial_balance

    # Start WebSocket manager if enabled
    if USE_WEBSOCKETS and ws_manager:
        await ws_manager.start()

    # Main bot loop
    while True:
        try:
            # System health check
            if not system_health_check():
                log_error("System health check failed. Pausing for 5 minutes.")
                await asyncio.sleep(300)
                continue

            # Get current account state
            current_balance = get_balance()

            # Update highest balance if current balance is higher
            if current_balance > highest_balance:
                highest_balance = current_balance
                log_info(f"New peak balance: ${highest_balance:.2f} USDT")

            # Check drawdown conditions
            continue_trading, current_drawdown, risk_multiplier = check_drawdown(initial_balance, highest_balance)

            log_info(f"Current Balance: ${current_balance:.2f} USDT | " +
                     f"Peak: ${highest_balance:.2f} | Drawdown: {current_drawdown:.2%}")

            if not continue_trading:
                log_error("Maximum drawdown reached. Trading halted.")
                break

            # Process each symbol
            for symbol in FUTURES_SYMBOLS:
                log_info(f"Processing {symbol}...")

                # Get market data (current and additional timeframes)
                df_main = get_historical_data(symbol)
                if df_main is None or len(df_main) < 200:
                    log_error(f"Insufficient data for {symbol}, skipping")
                    continue

                # Get additional timeframes for multi-timeframe analysis
                df_short = get_historical_data(symbol, interval="15m")
                df_long = get_historical_data(symbol, interval="4h")

                current_price = df_main['close'].iloc[-1]

                # Get position information
                position = get_position_info(symbol)

                # If already in position, manage it
                if position and abs(position['amount']) > 0:
                    log_info(f"Current {position['side']} Position: {position['amount']} contracts | " +
                             f"Entry: ${position['entry_price']:.2f} | P&L: {position['pnl_pct']:.2f}%")

                    position_closed = manage_open_position(symbol, position, df_main)

                    # Skip rest of loop for this symbol if already in position
                    if not position_closed:
                        continue

                # Analyze market conditions
                market_conditions = analyze_market_conditions(df_main)
                market_regime = detect_market_regime(df_main)

                log_info(f"Market: {market_regime} | Volatility: {market_conditions['volatility']:.4f} | " +
                         f"Trend Strength: {market_conditions['trend_strength']:.2f}")

                # Adjust risk based on market conditions
                market_risk_multiplier = adjust_risk_based_on_market(
                    market_conditions, market_regime, risk_multiplier
                )

                # Get multi-timeframe trade signal with score
                trade_signal, reason, score = should_enter_trade(
                    df_main=df_main,
                    df_short=df_short,
                    df_long=df_long,
                    market_conditions=market_conditions,
                    market_regime=market_regime
                )

                log_info(f"Signal: {trade_signal if trade_signal else 'HOLD'} | " +
                         f"Score: {score:.1f}/100 | Reason: {reason}")

                # Execute trade if we have a signal
                if trade_signal and score >= 70:  # Minimum score threshold
                    execute_new_trade(
                        symbol=symbol,
                        trade_signal=trade_signal,
                        reason=reason,
                        current_price=current_price,
                        market_data=df_main,
                        risk_multiplier=market_risk_multiplier,
                        score=score
                    )

            # Log daily statistics at midnight
            now = datetime.now()
            if now.hour == 0 and now.minute < 5:
                log_daily_statistics(trade_history)

            # Wait before next iteration
            log_info("Waiting for next cycle...")
            await asyncio.sleep(30)

        except KeyboardInterrupt:
            log_info("Bot stopped manually")
            break
        except Exception as e:
            log_error(f"Unexpected error: {str(e)}")
            log_error(traceback.format_exc())
            await asyncio.sleep(5)

    # Cleanup on exit
    if USE_WEBSOCKETS and ws_manager:
        await ws_manager.stop()

    log_info("Bot shutdown complete")


def run_bot():
    """Main entry point with asyncio loop handling"""
    if BACKTEST_MODE:
        print("Running in BACKTEST mode")
        from src.backtest import run_backtest
        run_backtest(FUTURES_SYMBOLS)
    else:
        # Create and run event loop for async operation
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(run_bot_async())
        finally:
            loop.close()


if __name__ == "__main__":
    run_bot()