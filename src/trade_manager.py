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
def calculate_position_size(symbol, entry_price, stop_loss_price):
    balance = get_balance()
    risk_amount = balance * RISK_PER_TRADE  # 1% of account balance
    stop_loss_diff = abs(entry_price - stop_loss_price)
    
    if stop_loss_diff == 0:
        return 0  # Prevent division by zero

    position_size = risk_amount / stop_loss_diff  # Position size based on stop-loss
    return round(position_size, 4)

# ✅ Place Market Order with Stop-Loss & Take-Profit
def place_order(symbol, side, stop_loss=None, take_profit=None):
    try:
        entry_price = float(client.futures_mark_price(symbol=symbol)["markPrice"])  # Get current price
        position_size = calculate_position_size(symbol, entry_price, stop_loss)

        if position_size <= 0:
            print(f"⚠️ Position size is too small. Skipping trade.")
            return None

        # ✅ Execute Market Order
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=position_size
        )

        # ✅ Set Stop-Loss & Take-Profit
        if stop_loss:
            client.futures_create_order(
                symbol=symbol,
                side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
                type=ORDER_TYPE_STOP_MARKET,
                stopPrice=stop_loss,
                closePosition=True
            )

        if take_profit:
            client.futures_create_order(
                symbol=symbol,
                side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
                type=ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=take_profit,
                closePosition=True
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
