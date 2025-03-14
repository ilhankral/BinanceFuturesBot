import os

# ✅ API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')

# ✅ Trading Strategy Selection
# Options: "EMA_CROSSOVER", "MACD_TREND", "RSI_OVERBOUGHT_OVERSOLD"
TRADING_STRATEGY = "EMA_CROSSOVER"  # Choose one strategy

# ✅ Futures Trading Configuration
FUTURES_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]  # Add your trading pairs

# ✅ Risk Management Settings
RISK_PER_TRADE = 0.01  # Risk 1% of total balance per trade
DEFAULT_LEVERAGE = 5  # Default leverage for all trades
MAX_DRAWDOWN = -0.05  # Stop trading if account drops 5%

# ✅ Stop-Loss & Take-Profit Settings
INITIAL_STOP_LOSS_PCT = -0.02  # Stop-Loss at -2%
INITIAL_TAKE_PROFIT_PCT = 0.04  # Take-Profit at +4%

# ✅ Trailing Stop Settings
ENABLE_TRAILING_STOP = True  # Set to False to disable trailing stops
TRAILING_STOP_PERCENT = 0.005  # Move stop-loss up/down by 0.5% as price moves

# ✅ Advanced Trade Management
DCA_STEP_PERCENT = 0.015  # Add more every -1.5% drop
MAX_DCA_COUNT = 3  # Maximum times to DCA

# ✅ Logging & Debugging
ENABLE_LOGGING = True  # Set to False to disable logs