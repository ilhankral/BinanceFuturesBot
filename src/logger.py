import logging
import colorlog
import os
from datetime import datetime
from tabulate import tabulate

# ✅ Ensure logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# ✅ Setup colored logging format
log_colors = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red'
}

formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors=log_colors
)

# ✅ Setup logger
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ✅ Also log to file
file_handler = logging.FileHandler(f"logs/trading_log_{datetime.now().strftime('%Y-%m-%d')}.log")
file_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(file_handler)

# ✅ Log Trade Execution
def log_trade(symbol, side, quantity, price, stop_loss, take_profit, status):
    log_message = [
        ["Symbol", symbol],
        ["Side", side],
        ["Quantity", quantity],
        ["Entry Price", f"${price:.2f}"],
        ["Stop-Loss", f"${stop_loss:.2f}"],
        ["Take-Profit", f"${take_profit:.2f}"],
        ["Status", status]
    ]
    formatted_log = tabulate(log_message, tablefmt="fancy_grid")

    logger.info(f"\n{formatted_log}")

# ✅ Log Trade Exits
def log_trade_exit(symbol, side, price, reason):
    log_message = f"✅ Trade Closed: {symbol} | Side: {side} | Exit Price: ${price:.2f} | Reason: {reason}"
    logger.info(log_message)

# ✅ Log Errors
def log_error(error_message):
    logger.error(f"❌ ERROR: {error_message}")
