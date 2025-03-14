from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_SECRET_KEY, RISK_PER_TRADE, DEFAULT_LEVERAGE, ENABLE_TRAILING_STOP, TRAILING_STOP_PERCENT
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL

# ✅ Define Futures Order Types
ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
ORDER_TYPE_TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"

# ✅ Initialize Binance Futures Client
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# ✅ Set Leverage for Futures
def set_leverage(symbol):
    client.futures_change_leverage(symbol=symbol, leverage=DEFAULT_LEVERAGE)

# ✅ Get USDT Balance
def get_balance():
    balance_info = client.futures_account_balance()
    for asset in balance_info:
        if asset["asset"] == "USDT":
            return float(asset["balance"])
    return 0.0

# ✅ Calculate Position Size Based on Risk
def calculate_position_size(symbol, entry_price, stop_loss_price, risk_multiplier=1.0):
    try:
        balance = get_balance()
        risk_amount = balance * RISK_PER_TRADE * risk_multiplier  # Apply drawdown-based risk adjustment

        # Get symbol info for precision
        symbol_info = client.futures_exchange_info()
        symbol_data = next((s for s in symbol_info['symbols'] if s['symbol'] == symbol), None)
        
        if not symbol_data:
            return 0

        # Get quantity precision
        quantity_precision = next(f for f in symbol_data['filters'] if f['filterType'] == 'LOT_SIZE')
        min_qty = float(quantity_precision['minQty'])
        step_size = float(quantity_precision['stepSize'])

        # Calculate position size based on risk
        stop_loss_diff = abs(entry_price - stop_loss_price)
        if stop_loss_diff == 0:
            return 0

        # Calculate raw position size
        position_size = risk_amount / stop_loss_diff
        
        # Apply leverage
        position_size = position_size * DEFAULT_LEVERAGE

        # Round to step size
        position_size = round(position_size / step_size) * step_size

        # Ensure minimum quantity
        if position_size < min_qty:
            return 0

        # Get current price for final check
        mark_price = float(client.futures_mark_price(symbol=symbol)['markPrice'])
        position_value = position_size * mark_price

        # Ensure position value doesn't exceed account risk limits
        max_position_value = balance * DEFAULT_LEVERAGE * 0.95 * risk_multiplier  # Apply risk multiplier to max position
        if position_value > max_position_value:
            position_size = (max_position_value / mark_price)
            position_size = round(position_size / step_size) * step_size

        return position_size

    except Exception as e:
        print(f"Error calculating position size: {e}")
        return 0

# ✅ Place Market Order with Stop-Loss & Take-Profit
def place_order(symbol, side, stop_loss=None, take_profit=None, risk_multiplier=1.0):
    try:
        # Get current market conditions
        mark_price = float(client.futures_mark_price(symbol=symbol)["markPrice"])
        
        # Get 24h price change
        ticker_24h = client.futures_ticker(symbol=symbol)
        price_change_24h = float(ticker_24h['priceChangePercent'])

        # Don't trade if market is too volatile
        if abs(price_change_24h) > 15:  # 15% change in 24h
            print(f"Market too volatile ({price_change_24h}% 24h change). Skipping trade.")
            return None

        position_size = calculate_position_size(symbol, mark_price, stop_loss, risk_multiplier)

        if position_size <= 0:
            print(f"⚠️ Position size too small or invalid. Skipping trade.")
            return None

        # Check if we have enough margin
        account = client.futures_account()
        available_balance = float(account['availableBalance'])
        required_margin = (position_size * mark_price) / DEFAULT_LEVERAGE

        if required_margin > available_balance * 0.95:  # Keep 5% margin buffer
            print(f"⚠️ Insufficient margin available. Required: {required_margin}, Available: {available_balance}")
            return None

        # Execute Market Order with reduced slippage
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=position_size,
            reduceOnly=False,
            newOrderRespType='RESULT'  # Get filled price
        )

        # Get actual fill price
        avg_price = float(order['avgPrice'])

        # Adjust stop-loss and take-profit based on fill price
        if stop_loss:
            stop_loss = round(avg_price * (1 - INITIAL_STOP_LOSS_PCT if side == SIDE_BUY else 1 + INITIAL_STOP_LOSS_PCT), 2)
            
        if take_profit:
            take_profit = round(avg_price * (1 + INITIAL_TAKE_PROFIT_PCT if side == SIDE_BUY else 1 - INITIAL_TAKE_PROFIT_PCT), 2)

        # Place stop-loss order
        if stop_loss:
            client.futures_create_order(
                symbol=symbol,
                side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
                type=ORDER_TYPE_STOP_MARKET,
                stopPrice=stop_loss,
                closePosition=True,
                workingType='MARK_PRICE'  # Use mark price for stops
            )

        # Place take-profit order
        if take_profit:
            client.futures_create_order(
                symbol=symbol,
                side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
                type=ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=take_profit,
                closePosition=True,
                workingType='MARK_PRICE'  # Use mark price for take-profits
            )

        return order

    except Exception as e:
        print(f"❌ Order Error: {e}")
        return None

# ✅ Update Stop-Loss Dynamically (Trailing Stop)
def update_trailing_stop(symbol, entry_price, trade_type):
    if not ENABLE_TRAILING_STOP:
        return

    current_price = float(client.futures_mark_price(symbol=symbol)["markPrice"])

    if trade_type == "LONG":
        new_stop_loss = round(current_price * (1 - TRAILING_STOP_PERCENT), 2)
        if new_stop_loss > entry_price:
            print(f"🔹 Adjusting Stop-Loss to {new_stop_loss} for LONG {symbol}")
            client.futures_create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_STOP_MARKET,
                stopPrice=new_stop_loss,
                closePosition=True
            )

    elif trade_type == "SHORT":
        new_stop_loss = round(current_price * (1 + TRAILING_STOP_PERCENT), 2)
        if new_stop_loss < entry_price:
            print(f"🔹 Adjusting Stop-Loss to {new_stop_loss} for SHORT {symbol}")
            client.futures_create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_STOP_MARKET,
                stopPrice=new_stop_loss,
                closePosition=True
            )
