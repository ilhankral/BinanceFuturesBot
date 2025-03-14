import time
import pandas as pd
from src.config import FUTURES_SYMBOLS
from src.trade_manager import place_order, set_leverage
from src.strategies import should_enter_trade, should_exit_trade
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL
from src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY
from logger import log_trade, log_trade_exit, log_error

# ✅ Initialize Binance Futures Client
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# ✅ Set leverage for all trading pairs
for symbol in FUTURES_SYMBOLS:
    set_leverage(symbol)

# ✅ Fetch account balance
def get_balance():
    balance_info = client.futures_account_balance()
    for asset in balance_info:
        if asset["asset"] == "USDT":
            return float(asset["balance"])
    return 0.0

# ✅ Fetch historical data
def get_historical_data(symbol):
    print(f"📊 Fetching market data for {symbol}...")
    klines = client.futures_klines(symbol=symbol, interval="1h", limit=300)
    df = pd.DataFrame(klines, columns=["time", "open", "high", "low", "close", "volume",
                                       "close_time", "quote_asset", "trades", "taker_base",
                                       "taker_quote", "ignore"])
    df["close"] = df["close"].astype(float)
    return df

# ✅ Main Trading Loop
def run_bot():
    print("🚀 Binance Futures Trading Bot Started 🚀")

    open_positions = {}  # Track open positions

    while True:
        try:
            account_balance = get_balance()
            print(f"\n💰 Current Balance: ${account_balance:.2f} USDT")

            for symbol in FUTURES_SYMBOLS:
                print(f"\n🔍 Analyzing {symbol}...")
                df = get_historical_data(symbol)
                trade_signal, reason = should_enter_trade(df)
                current_price = df.iloc[-1]["close"]

                print(f"📈 Latest Price: ${current_price:.2f}")
                print(f"📊 Signal: {trade_signal if trade_signal else 'HOLD'} | Reason: {reason}")

                # 📌 Check if we should enter a trade
                if trade_signal and symbol not in open_positions:
                    print(f"🔹 Trade Decision: {trade_signal} for {symbol}")

                    # 📌 Define Stop-Loss & Take-Profit Levels
                    stop_loss = round(current_price * (1 - 0.02 if trade_signal == "LONG" else 1 + 0.02), 2)
                    take_profit = round(current_price * (1 + 0.04 if trade_signal == "LONG" else 1 - 0.04), 2)

                    # 📌 Execute Trade
                    order = place_order(symbol, SIDE_BUY if trade_signal == "LONG" else SIDE_SELL,
                                        quantity=0.01, stop_loss=stop_loss, take_profit=take_profit)

                    if order:
                        log_trade(symbol, trade_signal, 0.01, current_price, stop_loss, take_profit, "EXECUTED")
                        open_positions[symbol] = {
                            "side": trade_signal,
                            "entry_price": current_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit
                        }

                # 📌 Check if we should exit a trade
                elif symbol in open_positions:
                    position = open_positions[symbol]
                    exit_trade, exit_reason = should_exit_trade(position["entry_price"], current_price)

                    if exit_trade:
                        print(f"✅ Closing Trade: {symbol} | Reason: {exit_reason}")
                        log_trade_exit(symbol, position["side"], current_price, exit_reason)
                        place_order(symbol, SIDE_SELL if position["side"] == "LONG" else SIDE_BUY, quantity=0.01)
                        del open_positions[symbol]

            print("\nWait For 1 Minute\n")
            time.sleep(60)  # 📌 Check every 60 seconds (1 minute)

        except KeyboardInterrupt:
            print("\n🛑 Bot Stopped Manually.")
            break
        except Exception as e:
            log_error(str(e))
            print(f"❌ Error: {e}")
            break

# ✅ Start the bot
if __name__ == "__main__":
    run_bot()
