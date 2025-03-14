"""
Enhanced Configuration Module for Binance Futures Trading Bot
- Comprehensive settings with validation
- Feature toggles
- Advanced strategy configuration
- Risk management controls
"""

import os
import json
from typing import Dict, List, Any
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Check for config file
CONFIG_FILE = os.getenv('CONFIG_FILE', 'config.json')
config_from_file = {}

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, 'r') as f:
            config_from_file = json.load(f)
        print(f"✅ Loaded configuration from {CONFIG_FILE}")
    except Exception as e:
        print(f"❌ Error loading config file: {e}")

# ================= API Configuration =================
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', config_from_file.get('BINANCE_API_KEY', ''))
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', config_from_file.get('BINANCE_SECRET_KEY', ''))

# ================= Mode Settings =================
# Enable/disable features and testing modes
BACKTEST_MODE = os.getenv('BACKTEST_MODE', 'false').lower() == 'true'
PAPER_TRADING = os.getenv('PAPER_TRADING', 'false').lower() == 'true'
USE_WEBSOCKETS = os.getenv('USE_WEBSOCKETS', 'true').lower() == 'true'
ENABLE_NOTIFICATIONS = os.getenv('ENABLE_NOTIFICATIONS', 'false').lower() == 'true'

# ================= Symbol Configuration =================
# Trading pairs to monitor and trade
FUTURES_SYMBOLS = config_from_file.get('FUTURES_SYMBOLS', ["BTCUSDT"])
if os.getenv('FUTURES_SYMBOLS'):
    FUTURES_SYMBOLS = os.getenv('FUTURES_SYMBOLS').split(',')

# Symbol-specific configuration (leverage, strategies, etc.)
SYMBOL_CONFIG = config_from_file.get('SYMBOL_CONFIG', {})
for symbol in FUTURES_SYMBOLS:
    if symbol not in SYMBOL_CONFIG:
        SYMBOL_CONFIG[symbol] = {
            "leverage": DEFAULT_LEVERAGE,
            "enabled_strategies": TRADING_STRATEGIES,
            "custom_stop_loss": None,  # Default to global setting
            "custom_take_profit": None  # Default to global setting
        }

# ================= Timeframe Configuration =================
# Main timeframe for analysis
MAIN_TIMEFRAME = os.getenv('MAIN_TIMEFRAME', config_from_file.get('MAIN_TIMEFRAME', '1h'))

# Additional timeframes for multi-timeframe analysis
ADDITIONAL_TIMEFRAMES = config_from_file.get('ADDITIONAL_TIMEFRAMES', ['15m', '4h', '1d'])
if os.getenv('ADDITIONAL_TIMEFRAMES'):
    ADDITIONAL_TIMEFRAMES = os.getenv('ADDITIONAL_TIMEFRAMES').split(',')

# ================= Strategy Configuration =================
# List of strategies to use (minimum 2 for confirmation)
TRADING_STRATEGIES = config_from_file.get('TRADING_STRATEGIES', [
    "EMA_CROSSOVER",
    "MACD_TREND",
    "RSI_OVERBOUGHT_OVERSOLD"
])
if os.getenv('TRADING_STRATEGIES'):
    TRADING_STRATEGIES = os.getenv('TRADING_STRATEGIES').split(',')

# Strategy weighting (importance of each strategy in decision)
STRATEGY_WEIGHTS = config_from_file.get('STRATEGY_WEIGHTS', {
    "EMA_CROSSOVER": 1.0,
    "MACD_TREND": 1.0,
    "RSI_OVERBOUGHT_OVERSOLD": 1.0,
    "BOLLINGER_BANDS": 0.8,
    "ICHIMOKU_CLOUD": 0.8,
    "VOLUME_PROFILE": 0.7
})

# Strategy parameters
STRATEGY_PARAMS = config_from_file.get('STRATEGY_PARAMS', {
    "EMA_CROSSOVER": {
        "fast_period": 50,
        "slow_period": 200
    },
    "MACD_TREND": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    },
    "RSI_OVERBOUGHT_OVERSOLD": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
    },
    "BOLLINGER_BANDS": {
        "period": 20,
        "std_dev": 2.0
    }
})

# Minimum strategy confirmation count required
MIN_STRATEGY_CONFIRMATION = int(os.getenv('MIN_STRATEGY_CONFIRMATION',
                                          config_from_file.get('MIN_STRATEGY_CONFIRMATION', 2)))

# Signal strength threshold (0-100) for trade execution
SIGNAL_THRESHOLD = float(os.getenv('SIGNAL_THRESHOLD',
                                   config_from_file.get('SIGNAL_THRESHOLD', 70)))

# ================= Risk Management Settings =================
# Maximum percentage of account to risk per trade
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE',
                                 config_from_file.get('RISK_PER_TRADE', 0.01)))  # 1% by default

# Default leverage for futures trading
DEFAULT_LEVERAGE = int(os.getenv('DEFAULT_LEVERAGE',
                                 config_from_file.get('DEFAULT_LEVERAGE', 5)))

# Maximum account drawdown before stopping (as a negative percentage)
MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN',
                               config_from_file.get('MAX_DRAWDOWN', -0.05)))  # -5% by default

# Maximum position size as percentage of account
MAX_POSITION_SIZE_PCT = float(os.getenv('MAX_POSITION_SIZE_PCT',
                                        config_from_file.get('MAX_POSITION_SIZE_PCT', 0.2)))  # 20% max

# Maximum number of simultaneous positions
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS',
                              config_from_file.get('MAX_POSITIONS', len(FUTURES_SYMBOLS))))

# Risk adjustment for different market regimes
MARKET_REGIME_RISK_ADJUSTMENTS = config_from_file.get('MARKET_REGIME_RISK_ADJUSTMENTS', {
    "trending": 1.0,  # Normal risk in trending markets
    "ranging": 0.7,  # Reduced risk in ranging markets
    "volatile": 0.5,  # Significantly reduced risk in volatile markets
    "choppy": 0.3,  # Minimal risk in choppy markets
    "unknown": 0.5  # Default to reduced risk
})

# ================= Stop-Loss & Take-Profit Settings =================
# Default stop-loss percentage (negative value)
INITIAL_STOP_LOSS_PCT = float(os.getenv('INITIAL_STOP_LOSS_PCT',
                                        config_from_file.get('INITIAL_STOP_LOSS_PCT', -0.02)))

# Default take-profit percentage (positive value)
INITIAL_TAKE_PROFIT_PCT = float(os.getenv('INITIAL_TAKE_PROFIT_PCT',
                                          config_from_file.get('INITIAL_TAKE_PROFIT_PCT', 0.04)))

# ================= Trailing Stop Settings =================
# Enable/disable trailing stops
ENABLE_TRAILING_STOP = os.getenv('ENABLE_TRAILING_STOP', 'true').lower() == 'true'
if 'ENABLE_TRAILING_STOP' in config_from_file:
    ENABLE_TRAILING_STOP = config_from_file['ENABLE_TRAILING_STOP']

# Trailing stop activation threshold (% profit needed to activate)
TRAILING_ACTIVATION_PCT = float(os.getenv('TRAILING_ACTIVATION_PCT',
                                          config_from_file.get('TRAILING_ACTIVATION_PCT', 0.01)))  # 1%

# Trailing stop percentage
TRAILING_STOP_PERCENT = float(os.getenv('TRAILING_STOP_PERCENT',
                                        config_from_file.get('TRAILING_STOP_PERCENT', 0.005)))  # 0.5%

# ================= Advanced Trade Management =================
# Enable/disable partial take-profit levels
ENABLE_PARTIAL_TAKE_PROFIT = os.getenv('ENABLE_PARTIAL_TAKE_PROFIT', 'false').lower() == 'true'
if 'ENABLE_PARTIAL_TAKE_PROFIT' in config_from_file:
    ENABLE_PARTIAL_TAKE_PROFIT = config_from_file['ENABLE_PARTIAL_TAKE_PROFIT']

# Partial take-profit levels (% of position to close at each profit level)
PARTIAL_TP_LEVELS = config_from_file.get('PARTIAL_TP_LEVELS', [
    {"profit_pct": 0.02, "close_pct": 0.3},  # Close 30% at 2% profit
    {"profit_pct": 0.03, "close_pct": 0.3},  # Close 30% at 3% profit
    {"profit_pct": 0.05, "close_pct": 0.4}  # Close remaining 40% at 5% profit
])

# Enable/disable dollar cost averaging on drawdown
ENABLE_DCA = os.getenv('ENABLE_DCA', 'false').lower() == 'true'
if 'ENABLE_DCA' in config_from_file:
    ENABLE_DCA = config_from_file['ENABLE_DCA']

# DCA settings
DCA_STEP_PERCENT = float(os.getenv('DCA_STEP_PERCENT',
                                   config_from_file.get('DCA_STEP_PERCENT', 0.015)))  # Add at 1.5% drop
MAX_DCA_COUNT = int(os.getenv('MAX_DCA_COUNT',
                              config_from_file.get('MAX_DCA_COUNT', 3)))  # Maximum times to DCA

# Trade timeout (hours) - close position if held too long
TRADE_TIMEOUT_HOURS = float(os.getenv('TRADE_TIMEOUT_HOURS',
                                      config_from_file.get('TRADE_TIMEOUT_HOURS', 72)))  # 3 days max

# ================= Operational Settings =================
# Polling interval (seconds) for REST API mode
REST_POLLING_INTERVAL = int(os.getenv('REST_POLLING_INTERVAL',
                                      config_from_file.get('REST_POLLING_INTERVAL', 30)))

# Lookback period for historical data (number of candles)
LOOKBACK_PERIOD = int(os.getenv('LOOKBACK_PERIOD',
                                config_from_file.get('LOOKBACK_PERIOD', 500)))

# ================= Notification Settings =================
# Telegram bot token (if enabled)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', config_from_file.get('TELEGRAM_BOT_TOKEN', ''))
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', config_from_file.get('TELEGRAM_CHAT_ID', ''))

# Discord webhook URL (if enabled)
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', config_from_file.get('DISCORD_WEBHOOK_URL', ''))

# ================= Logging Settings =================
# Enable/disable detailed logging
ENABLE_LOGGING = os.getenv('ENABLE_LOGGING', 'true').lower() == 'true'
if 'ENABLE_LOGGING' in config_from_file:
    ENABLE_LOGGING = config_from_file['ENABLE_LOGGING']

# Log file path
LOG_FILE = os.getenv('LOG_FILE', config_from_file.get('LOG_FILE', 'logs/trade_logs.txt'))

# Logging level
LOG_LEVEL = os.getenv('LOG_LEVEL', config_from_file.get('LOG_LEVEL', 'INFO'))
LOG_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
LOGGING_LEVEL = LOG_LEVEL_MAP.get(LOG_LEVEL.upper(), logging.INFO)

# ================= Performance Analytics Settings =================
# Enable/disable performance tracking
ENABLE_PERFORMANCE_ANALYTICS = os.getenv('ENABLE_PERFORMANCE_ANALYTICS', 'true').lower() == 'true'
if 'ENABLE_PERFORMANCE_ANALYTICS' in config_from_file:
    ENABLE_PERFORMANCE_ANALYTICS = config_from_file['ENABLE_PERFORMANCE_ANALYTICS']

# Performance data file
PERFORMANCE_DATA_FILE = os.getenv('PERFORMANCE_DATA_FILE',
                                  config_from_file.get('PERFORMANCE_DATA_FILE', 'data/performance.json'))


# ================= Configuration Validation =================
def validate_config():
    """Validate configuration settings and return errors if any"""
    errors = []

    # Check API keys if not in backtest mode
    if not BACKTEST_MODE and not PAPER_TRADING:
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            errors.append("API keys are required for live trading")

    # Check risk settings
    if RISK_PER_TRADE <= 0 or RISK_PER_TRADE > 0.1:
        errors.append(f"RISK_PER_TRADE should be between 0 and 0.1 (10%), got {RISK_PER_TRADE}")

    # Check strategy settings
    if len(TRADING_STRATEGIES) < MIN_STRATEGY_CONFIRMATION:
        errors.append(
            f"Not enough strategies ({len(TRADING_STRATEGIES)}) to meet minimum confirmation requirement ({MIN_STRATEGY_CONFIRMATION})")

    # Check for missing strategy parameters
    for strategy in TRADING_STRATEGIES:
        if strategy not in STRATEGY_PARAMS:
            errors.append(f"Missing parameters for strategy: {strategy}")

    # Ensure directories exist
    try:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        perf_dir = os.path.dirname(PERFORMANCE_DATA_FILE)
        if perf_dir and not os.path.exists(perf_dir):
            os.makedirs(perf_dir)
    except Exception as e:
        errors.append(f"Error creating directories: {str(e)}")

    # Return validation results
    if errors:
        return False, errors
    return True, []


# Run validation
CONFIG_VALID, CONFIG_ERRORS = validate_config()

# Print configuration status
if CONFIG_VALID:
    print("✅ Configuration validation successful")
else:
    print("❌ Configuration validation failed:")
    for error in CONFIG_ERRORS:
        print(f"  - {error}")


# Export configuration as dictionary (for saving/loading)
def export_config() -> Dict[str, Any]:
    """Export current configuration as a dictionary"""
    config = {
        # Only include settings that should be persisted
        "FUTURES_SYMBOLS": FUTURES_SYMBOLS,
        "SYMBOL_CONFIG": SYMBOL_CONFIG,
        "MAIN_TIMEFRAME": MAIN_TIMEFRAME,
        "ADDITIONAL_TIMEFRAMES": ADDITIONAL_TIMEFRAMES,
        "TRADING_STRATEGIES": TRADING_STRATEGIES,
        "STRATEGY_WEIGHTS": STRATEGY_WEIGHTS,
        "STRATEGY_PARAMS": STRATEGY_PARAMS,
        "MIN_STRATEGY_CONFIRMATION": MIN_STRATEGY_CONFIRMATION,
        "SIGNAL_THRESHOLD": SIGNAL_THRESHOLD,
        "RISK_PER_TRADE": RISK_PER_TRADE,
        "DEFAULT_LEVERAGE": DEFAULT_LEVERAGE,
        "MAX_DRAWDOWN": MAX_DRAWDOWN,
        "MAX_POSITION_SIZE_PCT": MAX_POSITION_SIZE_PCT,
        "MAX_POSITIONS": MAX_POSITIONS,
        "MARKET_REGIME_RISK_ADJUSTMENTS": MARKET_REGIME_RISK_ADJUSTMENTS,
        "INITIAL_STOP_LOSS_PCT": INITIAL_STOP_LOSS_PCT,
        "INITIAL_TAKE_PROFIT_PCT": INITIAL_TAKE_PROFIT_PCT,
        "ENABLE_TRAILING_STOP": ENABLE_TRAILING_STOP,
        "TRAILING_ACTIVATION_PCT": TRAILING_ACTIVATION_PCT,
        "TRAILING_STOP_PERCENT": TRAILING_STOP_PERCENT,
        "ENABLE_PARTIAL_TAKE_PROFIT": ENABLE_PARTIAL_TAKE_PROFIT,
        "PARTIAL_TP_LEVELS": PARTIAL_TP_LEVELS,
        "ENABLE_DCA": ENABLE_DCA,
        "DCA_STEP_PERCENT": DCA_STEP_PERCENT,
        "MAX_DCA_COUNT": MAX_DCA_COUNT,
        "TRADE_TIMEOUT_HOURS": TRADE_TIMEOUT_HOURS
    }
    return config


# Save current configuration to file
def save_config_to_file(filename: str = CONFIG_FILE) -> bool:
    """Save current configuration to a JSON file"""
    try:
        config = export_config()
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False