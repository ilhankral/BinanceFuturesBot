import logging
import os
from tabulate import tabulate
from config import LOG_FILE

# ✅ Ensure logs directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# ✅ Configure Logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ✅ Log Trade Execution
def log_trade(symbol, trade_type, quantity, entry_price, stop_loss, take_profit, status):
    log_message = f"TRADE EXECUTED | {symbol} | {trade_type} | Qty: {quantity} | Entry: {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f} | Status: {status}"
    logging.info(log_message)
    print(log_message)


# ✅ Log Trade Exit & Profit/Loss
def log_trade_exit(symbol, trade_type, exit_price, entry_price, reason):
    profit_loss = (exit_price - entry_price) if trade_type == "LONG" else (entry_price - exit_price)
    log_message = f"TRADE CLOSED | {symbol} | {trade_type} | Entry: {entry_price:.2f} | Exit: {exit_price:.2f} | P/L: {profit_loss:.2f} USDT | Reason: {reason}"

    logging.info(log_message)
    print(log_message)


# ✅ Log Errors
def log_error(message):
    logging.error(f"ERROR: {message}")
    print(f"❌ ERROR: {message}")


# ✅ Display Log Summary in Table Format
def show_trade_logs():
    if not os.path.exists(LOG_FILE):
        print("📁 No trade logs found.")
        return

    with open(LOG_FILE, "r") as f:
        logs = [line.strip().split(" | ") for line in f.readlines() if "TRADE" in line]

    if logs:
        headers = ["Timestamp", "Level", "Trade Type", "Symbol", "Qty", "Entry", "SL", "TP", "Status"]
        print(tabulate(logs, headers=headers, tablefmt="pretty"))
    else:
        print("📁 No trade logs yet.")