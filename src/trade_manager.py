"""
Enhanced Trade Manager Module
- Advanced order execution
- Position sizing and risk management
- Intelligent stop-loss and take-profit management
- Trailing stops and partial position exits
- DCA and position adjustment logic
"""

import time
import uuid
import hmac
import hashlib
import urllib.parse
import requests
import json
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL

# Import configuration
from src.config import (
    BINANCE_API_KEY, BINANCE_SECRET_KEY, DEFAULT_LEVERAGE, RISK_PER_TRADE,
    MAX_POSITION_SIZE_PCT, ENABLE_TRAILING_STOP, TRAILING_ACTIVATION_PCT,
    TRAILING_STOP_PERCENT, ENABLE_PARTIAL_TAKE_PROFIT, PARTIAL_TP_LEVELS,
    MARKET_REGIME_RISK_ADJUSTMENTS, PAPER_TRADING
)

# Import logger
from src.logger import log_trade, log_error, log_info

# Initialize Binance Futures Client
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# Track open orders (useful for paper trading and order management)
open_orders = {}  # order_id -> order_details
open_positions = {}  # symbol -> position_details
paper_balances = {'USDT': 10000.0}  # Default starting balance for paper trading


def round_step_size(quantity: float, step_size: float) -> float:
    """Round quantity to step size"""
    precision = len(str(step_size).split('.')[-1])
    if precision == 1 and str(step_size).split('.')[-1] == '0':
        return int(quantity)

    quantity = Decimal(str(quantity))
    return float(quantity.quantize(Decimal(str(step_size)), rounding=ROUND_DOWN))


def get_symbol_info(symbol: str) -> Optional[Dict]:
    """Get trading precision and limits for a symbol"""
    try:
        # Get exchange info
        exchange_info = client.futures_exchange_info()

        # Find symbol info
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                filters = {}
                for f in symbol_info['filters']:
                    filters[f['filterType']] = f

                return {
                    'symbol': symbol,
                    'baseAsset': symbol_info['baseAsset'],
                    'quoteAsset': symbol_info['quoteAsset'],
                    'pricePrecision': symbol_info['pricePrecision'],
                    'quantityPrecision': symbol_info['quantityPrecision'],
                    'filters': filters,
                    'lotSizeFilter': next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None),
                    'priceFilter': next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None),
                    'minQty': float(
                        next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), {'minQty': '0.001'})[
                            'minQty']),
                    'maxQty': float(
                        next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), {'maxQty': '10000'})[
                            'maxQty']),
                    'stepSize': float(next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'),
                                           {'stepSize': '0.001'})['stepSize']),
                    'minPrice': float(next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'),
                                           {'minPrice': '0.01'})['minPrice']),
                    'maxPrice': float(next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'),
                                           {'maxPrice': '1000000'})['maxPrice']),
                    'tickSize': float(next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'),
                                           {'tickSize': '0.01'})['tickSize'])
                }

        return None

    except Exception as e:
        log_error(f"Error getting symbol info for {symbol}: {e}")
        return None


def set_leverage(symbol: str, leverage: int = DEFAULT_LEVERAGE) -> bool:
    """Set leverage for a trading pair"""
    if PAPER_TRADING:
        # In paper trading mode, just log and return success
        log_info(f"Paper Trading: Leverage set to {leverage}x for {symbol}")
        return True

    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        log_info(f"Leverage set to {leverage}x for {symbol}")
        return True
    except Exception as e:
        log_error(f"Error setting leverage for {symbol}: {e}")
        return False


def get_balance(asset: str = 'USDT') -> float:
    """Get available balance of an asset"""
    if PAPER_TRADING:
        # Use paper trading balance
        return paper_balances.get(asset, 0.0)

    try:
        account_info = client.futures_account_balance()
        for balance in account_info:
            if balance['asset'] == asset:
                return float(balance['balance'])
        return 0.0
    except Exception as e:
        log_error(f"Error getting balance for {asset}: {e}")
        return 0.0


def get_mark_price(symbol: str) -> Optional[float]:
    """Get current mark price for a symbol"""
    try:
        mark_price_info = client.futures_mark_price(symbol=symbol)
        return float(mark_price_info['markPrice'])
    except Exception as e:
        log_error(f"Error getting mark price for {symbol}: {e}")
        return None


def adjust_risk_based_on_market(market_conditions: Dict, market_regime: str, base_risk_multiplier: float) -> float:
    """
    Adjust risk based on market conditions and regime

    Args:
        market_conditions: Dict with market metrics
        market_regime: String indicating the market regime
        base_risk_multiplier: Base risk value from drawdown protection

    Returns:
        float: Adjusted risk multiplier
    """
    try:
        # Get regime adjustment factor
        regime_factor = MARKET_REGIME_RISK_ADJUSTMENTS.get(market_regime, 0.5)

        # Get volatility adjustment factor
        volatility = market_conditions.get('volatility', 0.01)
        volatility_factor = 1.0

        if volatility > 0.03:  # Very high volatility
            volatility_factor = 0.5
        elif volatility > 0.02:  # High volatility
            volatility_factor = 0.75
        elif volatility < 0.005:  # Very low volatility
            volatility_factor = 1.2

        # Check for abnormal volume
        if market_conditions.get('abnormal_volume', False):
            volume_factor = 0.8  # Reduce risk on abnormal volume
        else:
            volume_factor = 1.0

        # Calculate combined risk adjustment
        adjusted_risk = base_risk_multiplier * regime_factor * volatility_factor * volume_factor

        # Ensure risk doesn't go too low or too high
        adjusted_risk = max(0.2, min(1.2, adjusted_risk))

        log_info(f"Risk adjustment: Base {base_risk_multiplier:.2f} -> Adjusted {adjusted_risk:.2f} " +
                 f"(Regime: {regime_factor:.2f}, Vol: {volatility_factor:.2f}, Vol: {volume_factor:.2f})")

        return adjusted_risk

    except Exception as e:
        log_error(f"Error adjusting risk: {e}")
        return base_risk_multiplier


def calculate_position_size(symbol: str, entry_price: float, stop_loss_price: float,
                            risk_multiplier: float = 1.0) -> Tuple[float, float]:
    """
    Calculate position size based on risk parameters

    Returns:
        Tuple of (quantity, usd_size)
    """
    try:
        # Get current available balance
        balance = get_balance('USDT')

        # Skip if balance is too low
        if balance <= 0:
            log_error(f"Insufficient balance: {balance} USDT")
            return 0.0, 0.0

        # Calculate risk amount in USDT
        risk_amount = balance * RISK_PER_TRADE * risk_multiplier

        # Get symbol information for precision
        symbol_info = get_symbol_info(symbol)
        if not symbol_info:
            log_error(f"Could not get symbol info for {symbol}")
            return 0.0, 0.0

        # Calculate stop loss percentage
        stop_loss_pct = abs(entry_price - stop_loss_price) / entry_price

        # Calculate position size without leverage
        if stop_loss_pct <= 0.001:  # Prevent division by zero or tiny values
            log_error(f"Stop loss too close to entry price: {stop_loss_pct:.4%}")
            return 0.0, 0.0

        # Calculate position size in quote currency (USDT)
        position_size_quote = risk_amount / stop_loss_pct

        # Apply leverage
        position_size_quote_with_leverage = position_size_quote * DEFAULT_LEVERAGE

        # Check against maximum position size (% of balance)
        max_position_size = balance * MAX_POSITION_SIZE_PCT * DEFAULT_LEVERAGE
        if position_size_quote_with_leverage > max_position_size:
            log_info(
                f"Position size {position_size_quote_with_leverage:.2f} USDT exceeds max {max_position_size:.2f} USDT. Capping.")
            position_size_quote_with_leverage = max_position_size

        # Convert to quantity using entry price
        quantity = position_size_quote_with_leverage / entry_price

        # Apply lot size filter
        step_size = symbol_info['stepSize']
        quantity = round_step_size(quantity, step_size)

        # Ensure minimum quantity
        if quantity < symbol_info['minQty']:
            log_info(f"Calculated quantity {quantity} below minimum {symbol_info['minQty']} for {symbol}")
            return 0.0, 0.0

        # Ensure maximum quantity
        if quantity > symbol_info['maxQty']:
            log_info(f"Calculated quantity {quantity} above maximum {symbol_info['maxQty']} for {symbol}")
            quantity = symbol_info['maxQty']

        # Recalculate actual USD size
        actual_usd_size = quantity * entry_price

        log_info(f"Position size calculation for {symbol}: " +
                 f"Risk {risk_amount:.2f} USDT ({RISK_PER_TRADE * 100:.1f}% * {risk_multiplier:.2f}), " +
                 f"SL {stop_loss_pct:.2%}, Qty {quantity}, Value {actual_usd_size:.2f} USDT")

        return quantity, actual_usd_size

    except Exception as e:
        log_error(f"Error calculating position size: {e}")
        return 0.0, 0.0


def round_price_to_tick(price: float, symbol: str) -> float:
    """Round price to valid tick size"""
    try:
        # Get symbol info
        symbol_info = get_symbol_info(symbol)
        if not symbol_info:
            return price

        # Get tick size
        tick_size = symbol_info['tickSize']

        # Round to tick size
        price_precision = len(str(tick_size).split('.')[-1])
        rounded_price = round(price / tick_size) * tick_size

        return float(f"{rounded_price:.{price_precision}f}")

    except Exception as e:
        log_error(f"Error rounding price: {e}")
        return price


def place_order(symbol: str, side: str, stop_loss: Optional[float] = None,
                take_profit: Optional[float] = None, risk_multiplier: float = 1.0,
                order_type: str = ORDER_TYPE_MARKET, limit_price: Optional[float] = None) -> Optional[Dict]:
    """
    Place a futures order with stop-loss and take-profit

    Returns:
        Dict with order details or None on failure
    """
    try:
        # Generate a unique client order ID
        client_order_id = f"bot_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        # Get current mark price if needed
        current_price = limit_price if limit_price else get_mark_price(symbol)
        if not current_price:
            log_error(f"Could not get price for {symbol}")
            return None

        # Determine entry price based on order type
        entry_price = limit_price if order_type != ORDER_TYPE_MARKET else current_price

        # Calculate appropriate stop-loss if not provided
        if not stop_loss:
            # Default stop-loss at 2% below/above entry
            if side == SIDE_BUY:
                stop_loss = round_price_to_tick(entry_price * 0.98, symbol)
            else:
                stop_loss = round_price_to_tick(entry_price * 1.02, symbol)

        # Calculate appropriate take-profit if not provided
        if not take_profit:
            # Default take-profit at 4% above/below entry
            if side == SIDE_BUY:
                take_profit = round_price_to_tick(entry_price * 1.04, symbol)
            else:
                take_profit = round_price_to_tick(entry_price * 0.96, symbol)

        # Calculate position size
        quantity, usd_size = calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            risk_multiplier=risk_multiplier
        )

        if quantity <= 0 or usd_size <= 0:
            log_error(f"Invalid position size calculated: {quantity} {symbol} ({usd_size} USD)")
            return None

        # Handle paper trading
        if PAPER_TRADING:
            paper_order = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': float(quantity),
                'price': entry_price,
                'stopPrice': stop_loss,
                'takeProfit': take_profit,
                'orderId': f"paper_{int(time.time() * 1000)}",
                'clientOrderId': client_order_id,
                'status': 'FILLED',
                'executedQty': float(quantity),
                'origQty': float(quantity),
                'avgPrice': entry_price,
                'time': int(time.time() * 1000),
                'paper': True
            }

            # Update paper trading balance
            margin_required = usd_size / DEFAULT_LEVERAGE
            paper_balances['USDT'] = max(0, paper_balances.get('USDT', 0) - margin_required)

            # Log the paper trade
            log_trade(
                symbol=symbol,
                trade_type="LONG" if side == SIDE_BUY else "SHORT",
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status="PAPER_TRADE"
            )

            # Track the position
            open_positions[symbol] = {
                'symbol': symbol,
                'side': "LONG" if side == SIDE_BUY else "SHORT",
                'entry_price': entry_price,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'usd_size': usd_size,
                'leverage': DEFAULT_LEVERAGE,
                'entry_time': datetime.now(),
                'paper': True
            }

            # Return the paper order
            return paper_order

        # Place real order on Binance
        order_params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': float(quantity),
            'newClientOrderId': client_order_id
        }

        # Add price for limit orders
        if order_type != ORDER_TYPE_MARKET and limit_price:
            order_params['price'] = limit_price

        # Execute order
        order = client.futures_create_order(**order_params)

        # Wait for order to execute (needed for market orders to get filled price)
        time.sleep(1)

        # Get order details
        order_details = client.futures_get_order(symbol=symbol, orderId=order['orderId'])

        # Check if order was filled
        if order_details['status'] != 'FILLED':
            log_error(f"Order not filled: {order_details}")
            return None

        # Get filled price
        filled_price = float(order_details['avgPrice'])

        # Adjust stop-loss and take-profit based on filled price
        if side == SIDE_BUY:
            adjusted_stop_loss = round_price_to_tick(filled_price * 0.98, symbol)
            adjusted_take_profit = round_price_to_tick(filled_price * 1.04, symbol)
        else:
            adjusted_stop_loss = round_price_to_tick(filled_price * 1.02, symbol)
            adjusted_take_profit = round_price_to_tick(filled_price * 0.96, symbol)

        # Place stop-loss order
        try:
            stop_loss_order = client.futures_create_order(
                symbol=symbol,
                side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
                type='STOP_MARKET',
                stopPrice=adjusted_stop_loss,
                closePosition=True,
                timeInForce='GTC'
            )

            # Store stop-loss order ID
            order_details['stopLossOrderId'] = stop_loss_order['orderId']

        except Exception as e:
            log_error(f"Error placing stop-loss order: {e}")

        # Place take-profit order
        try:
            take_profit_order = client.futures_create_order(
                symbol=symbol,
                side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
                type='TAKE_PROFIT_MARKET',
                stopPrice=adjusted_take_profit,
                closePosition=True,
                timeInForce='GTC'
            )

            # Store take-profit order ID
            order_details['takeProfitOrderId'] = take_profit_order['orderId']

        except Exception as e:
            log_error(f"Error placing take-profit order: {e}")

        # Handle partial take-profit levels if enabled
        if ENABLE_PARTIAL_TAKE_PROFIT:
            _place_partial_take_profit_orders(
                symbol=symbol,
                side=side,
                entry_price=filled_price,
                quantity=quantity
            )

        # Log the trade
        log_trade(
            symbol=symbol,
            trade_type="LONG" if side == SIDE_BUY else "SHORT",
            quantity=quantity,
            entry_price=filled_price,
            stop_loss=adjusted_stop_loss,
            take_profit=adjusted_take_profit,
            status="EXECUTED"
        )

        # Track the position
        open_positions[symbol] = {
            'symbol': symbol,
            'side': "LONG" if side == SIDE_BUY else "SHORT",
            'entry_price': filled_price,
            'quantity': quantity,
            'stop_loss': adjusted_stop_loss,
            'take_profit': adjusted_take_profit,
            'usd_size': quantity * filled_price,
            'leverage': DEFAULT_LEVERAGE,
            'entry_time': datetime.now(),
            'paper': False
        }

        return order_details

    except Exception as e:
        log_error(f"Error placing order: {e}")
        return None


def _place_partial_take_profit_orders(symbol: str, side: str, entry_price: float, quantity: float) -> None:
    """Place partial take-profit orders according to configuration"""
    if not ENABLE_PARTIAL_TAKE_PROFIT or not PARTIAL_TP_LEVELS:
        return

    try:
        # Sort levels by profit percentage
        sorted_levels = sorted(PARTIAL_TP_LEVELS, key=lambda x: x['profit_pct'])

        # Create counter side
        counter_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY

        for level in sorted_levels:
            profit_pct = level['profit_pct']
            close_pct = level['close_pct']

            # Calculate take-profit price
            if side == SIDE_BUY:
                tp_price = round_price_to_tick(entry_price * (1 + profit_pct), symbol)
            else:
                tp_price = round_price_to_tick(entry_price * (1 - profit_pct), symbol)

            # Calculate quantity for this level
            level_qty = quantity * close_pct

            # Round quantity
            symbol_info = get_symbol_info(symbol)
            if symbol_info:
                level_qty = round_step_size(level_qty, symbol_info['stepSize'])

            # Ensure minimum quantity
            if level_qty < (symbol_info['minQty'] if symbol_info else 0.001):
                continue

            # Place limit order for partial take-profit
            try:
                tp_order = client.futures_create_order(
                    symbol=symbol,
                    side=counter_side,
                    type='LIMIT',
                    price=tp_price,
                    quantity=level_qty,
                    timeInForce='GTC'
                )

                log_info(f"Placed partial take-profit: {symbol} {counter_side} {level_qty} @ {tp_price} " +
                         f"({profit_pct * 100:.1f}% profit, {close_pct * 100:.0f}% of position)")

            except Exception as e:
                log_error(f"Error placing partial take-profit: {e}")

    except Exception as e:
        log_error(f"Error setting up partial take-profits: {e}")


def update_trailing_stop(symbol: str, entry_price: float, current_price: float,
                         trade_type: str) -> bool:
    """
    Update trailing stop-loss based on price movement

    Args:
        symbol: Trading pair
        entry_price: Original entry price
        current_price: Current market price
        trade_type: 'LONG' or 'SHORT'

    Returns:
        bool: True if stop-loss was updated
    """
    if not ENABLE_TRAILING_STOP:
        return False

    try:
        # Calculate current profit percentage
        if trade_type == "LONG":
            profit_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            profit_pct = (entry_price - current_price) / entry_price

        # Only activate trailing stop if profit exceeds activation threshold
        if profit_pct < TRAILING_ACTIVATION_PCT:
            return False

        # Get current position
        position = open_positions.get(symbol)
        if not position:
            return False

        # Calculate new stop-loss level
        if trade_type == "LONG":
            new_stop_loss = round_price_to_tick(current_price * (1 - TRAILING_STOP_PERCENT), symbol)

            # Only move stop-loss if it would move up
            if new_stop_loss <= position.get('stop_loss', 0):
                return False

        else:  # SHORT
            new_stop_loss = round_price_to_tick(current_price * (1 + TRAILING_STOP_PERCENT), symbol)

            # Only move stop-loss if it would move down
            if new_stop_loss >= position.get('stop_loss', float('inf')):
                return False

        # Handle paper trading
        if PAPER_TRADING or position.get('paper', False):
            # Just update the stored position
            position['stop_loss'] = new_stop_loss
            log_info(f"Paper Trading: Updated trailing stop for {symbol} {trade_type} to {new_stop_loss}")
            return True

        # Find existing stop-loss order
        open_orders = client.futures_get_open_orders(symbol=symbol)
        stop_loss_order = next((o for o in open_orders if o['type'] in ['STOP_MARKET', 'STOP'] and
                                o['side'] == (SIDE_SELL if trade_type == "LONG" else SIDE_BUY)), None)

        # Cancel existing stop-loss order
        if stop_loss_order:
            client.futures_cancel_order(
                symbol=symbol,
                orderId=stop_loss_order['orderId']
            )

        # Place new stop-loss order
        new_stop_order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL if trade_type == "LONG" else SIDE_BUY,
            type='STOP_MARKET',
            stopPrice=new_stop_loss,
            closePosition=True,
            timeInForce='GTC'
        )

        # Update position record
        position['stop_loss'] = new_stop_loss

        log_info(f"Updated trailing stop for {symbol} {trade_type} to {new_stop_loss} " +
                 f"({profit_pct * 100:.2f}% profit)")

        return True

    except Exception as e:
        log_error(f"Error updating trailing stop: {e}")
        return False


def close_position(symbol: str, position_side: str, reason: str) -> bool:
    """
    Close an open position

    Args:
        symbol: Trading pair
        position_side: 'LONG' or 'SHORT'
        reason: Reason for closing

    Returns:
        bool: True if position was closed
    """
    try:
        # Handle paper trading
        position = open_positions.get(symbol)
        if PAPER_TRADING or (position and position.get('paper', False)):
            # Calculate profit/loss
            current_price = get_mark_price(symbol)
            if not current_price or not position:
                log_error(f"Cannot close paper position: missing price or position data")
                return False

            # Calculate PnL
            entry_price = position.get('entry_price', current_price)
            quantity = position.get('quantity', 0)

            if position_side == "LONG":
                pnl = (current_price - entry_price) * quantity
            else:  # SHORT
                pnl = (entry_price - current_price) * quantity

            # Update paper balance
            paper_balances['USDT'] = paper_balances.get('USDT', 0) + (quantity * current_price / DEFAULT_LEVERAGE) + pnl

            # Remove position
            if symbol in open_positions:
                del open_positions[symbol]

            log_info(f"Paper Trading: Closed {position_side} position for {symbol} at {current_price}. " +
                     f"PnL: {pnl:.2f} USDT. Reason: {reason}")

            return True

        # Place market order to close real position
        close_side = SIDE_SELL if position_side == "LONG" else SIDE_BUY

        # Get position details
        positions = client.futures_position_information(symbol=symbol)
        position_info = next((p for p in positions if float(p['positionAmt']) != 0), None)

        if not position_info:
            log_error(f"No open position found for {symbol}")
            return False

        # Calculate quantity to close
        quantity = abs(float(position_info['positionAmt']))

        # Place market order to close
        close_order = client.futures_create_order(
            symbol=symbol,
            side=close_side,
            type=ORDER_TYPE_MARKET,
            quantity=quantity,
            reduceOnly=True
        )

        # Cancel any open orders for the symbol
        open_orders = client.futures_get_open_orders(symbol=symbol)
        for order in open_orders:
            try:
                client.futures_cancel_order(
                    symbol=symbol,
                    orderId=order['orderId']
                )
            except:
                pass

        # Remove from open positions
        if symbol in open_positions:
            del open_positions[symbol]

        log_info(f"Closed {position_side} position for {symbol}. Quantity: {quantity}. Reason: {reason}")

        return True

    except Exception as e:
        log_error(f"Error closing position: {e}")
        return False


def add_to_position(symbol: str, side: str, additional_risk: float = 0.5) -> bool:
    """
    Add to an existing position (DCA)

    Args:
        symbol: Trading pair
        side: 'LONG' or 'SHORT'
        additional_risk: Fraction of standard risk to use

    Returns:
        bool: True if position was increased
    """
    try:
        # Get current position
        position = open_positions.get(symbol)
        if not position:
            return False

        # Verify side matches
        if position['side'] != ("LONG" if side == SIDE_BUY else "SHORT"):
            log_error(f"Cannot add to position: side mismatch")
            return False

        # Get current price
        current_price = get_mark_price(symbol)
        if not current_price:
            return False

        # Use existing stop-loss
        stop_loss = position.get('stop_loss')

        # Create order with reduced risk
        order = place_order(
            symbol=symbol,
            side=side,
            stop_loss=stop_loss,
            risk_multiplier=additional_risk
        )

        if not order:
            return False

        # Update average entry price
        if position.get('quantity', 0) > 0 and order.get('executedQty', 0) > 0:
            # Calculate new weighted average entry
            total_qty = position['quantity'] + float(order['executedQty'])
            new_entry = (position['entry_price'] * position['quantity'] +
                         float(order['avgPrice']) * float(order['executedQty'])) / total_qty

            # Update position details
            position['quantity'] = total_qty
            position['entry_price'] = new_entry
            position['usd_size'] = total_qty * new_entry

            log_info(f"Added to {position['side']} position for {symbol}. " +
                     f"New qty: {total_qty}, Avg entry: {new_entry}")

        return True

    except Exception as e:
        log_error(f"Error adding to position: {e}")
        return False


def adjust_take_profit(symbol: str, new_take_profit_pct: float) -> bool:
    """
    Adjust take-profit level for an open position

    Args:
        symbol: Trading pair
        new_take_profit_pct: New take-profit as percentage from entry

    Returns:
        bool: True if take-profit was adjusted
    """
    try:
        # Get current position
        position = open_positions.get(symbol)
        if not position:
            return False

        # Calculate new take-profit price
        entry_price = position['entry_price']
        if position['side'] == "LONG":
            new_tp_price = round_price_to_tick(entry_price * (1 + new_take_profit_pct), symbol)
        else:  # SHORT
            new_tp_price = round_price_to_tick(entry_price * (1 - new_take_profit_pct), symbol)

        # Handle paper trading
        if PAPER_TRADING or position.get('paper', False):
            position['take_profit'] = new_tp_price
            log_info(f"Paper Trading: Adjusted take-profit for {symbol} to {new_tp_price}")
            return True

        # Find existing take-profit order
        open_orders = client.futures_get_open_orders(symbol=symbol)
        tp_order = next((o for o in open_orders if o['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT'] and
                         o['side'] == (SIDE_SELL if position['side'] == "LONG" else SIDE_BUY)), None)

        # Cancel existing take-profit order
        if tp_order:
            client.futures_cancel_order(
                symbol=symbol,
                orderId=tp_order['orderId']
            )

        # Place new take-profit order
        new_tp_order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL if position['side'] == "LONG" else SIDE_BUY,
            type='TAKE_PROFIT_MARKET',
            stopPrice=new_tp_price,
            closePosition=True,
            timeInForce='GTC'
        )

        # Update position record
        position['take_profit'] = new_tp_price

        log_info(f"Adjusted take-profit for {symbol} to {new_tp_price} " +
                 f"({new_take_profit_pct * 100:.2f}% from entry)")

        return True

    except Exception as e:
        log_error(f"Error adjusting take-profit: {e}")
        return False


def move_stop_loss_to_breakeven(symbol: str) -> bool:
    """
    Move stop-loss to breakeven once profit reaches a threshold

    Args:
        symbol: Trading pair

    Returns:
        bool: True if stop-loss was moved
    """
    try:
        # Get current position
        position = open_positions.get(symbol)
        if not position:
            return False

        # Get current price
        current_price = get_mark_price(symbol)
        if not current_price:
            return False

        # Calculate current profit percentage
        if position['side'] == "LONG":
            profit_pct = (current_price - position['entry_price']) / position['entry_price']
        else:  # SHORT
            profit_pct = (position['entry_price'] - current_price) / position['entry_price']

        # Only move to breakeven if profit exceeds threshold (2%)
        if profit_pct < 0.02:
            return False

        # Calculate new stop-loss price (with small buffer)
        if position['side'] == "LONG":
            new_stop_loss = round_price_to_tick(position['entry_price'] * 1.001, symbol)  # 0.1% above entry
        else:  # SHORT
            new_stop_loss = round_price_to_tick(position['entry_price'] * 0.999, symbol)  # 0.1% below entry

        # Handle paper trading
        if PAPER_TRADING or position.get('paper', False):
            position['stop_loss'] = new_stop_loss
            log_info(f"Paper Trading: Moved stop-loss to breakeven for {symbol} at {new_stop_loss}")
            return True

        # Find existing stop-loss order
        open_orders = client.futures_get_open_orders(symbol=symbol)
        sl_order = next((o for o in open_orders if o['type'] in ['STOP_MARKET', 'STOP'] and
                         o['side'] == (SIDE_SELL if position['side'] == "LONG" else SIDE_BUY)), None)

        # Cancel existing stop-loss order
        if sl_order:
            client.futures_cancel_order(
                symbol=symbol,
                orderId=sl_order['orderId']
            )

        # Place new stop-loss order
        new_sl_order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL if position['side'] == "LONG" else SIDE_BUY,
            type='STOP_MARKET',
            stopPrice=new_stop_loss,
            closePosition=True,
            timeInForce='GTC'
        )

        # Update position record
        position['stop_loss'] = new_stop_loss

        log_info(f"Moved stop-loss to breakeven for {symbol} at {new_stop_loss}")

        return True

    except Exception as e:
        log_error(f"Error moving stop-loss to breakeven: {e}")
        return False