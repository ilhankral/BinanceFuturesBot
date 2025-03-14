import time
import pandas as pd
import numpy as np
from datetime import datetime
from src.config import FUTURES_SYMBOLS, MAX_DRAWDOWN
from src.trade_manager import place_order, set_leverage, calculate_position_size
from src.strategies import should_enter_trade, should_exit_trade, analyze_market_conditions
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL
from src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY
from logger import log_trade, log_trade_exit, log_error

# Initialize Binance Futures Client
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# Set leverage for all trading pairs
for symbol in FUTURES_SYMBOLS:
    set_leverage(symbol)

def get_balance():
    try:
        balance_info = client.futures_account_balance()
        for asset in balance_info:
            if asset["asset"] == "USDT":
                return float(asset["balance"])
        return 0.0
    except Exception as e:
        log_error(f"Error getting balance: {e}")
        return 0.0

def get_historical_data(symbol, interval="1h", limit=300):
    try:
        print(f"📊 Fetching market data for {symbol}...")
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=["time", "open", "high", "low", "close", "volume",
                                         "close_time", "quote_asset", "trades", "taker_base",
                                         "taker_quote", "ignore"])
        
        # Convert to numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])
            
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df
    except Exception as e:
        log_error(f"Error fetching data for {symbol}: {e}")
        return None

def check_drawdown(initial_balance):
    try:
        current_balance = get_balance()
        if current_balance <= 0:
            return False  # Stop trading if balance is zero or negative
        
        # Calculate drawdown percentage
        drawdown = (initial_balance - current_balance) / initial_balance
        
        # Check if drawdown exceeds maximum allowed
        if drawdown > abs(MAX_DRAWDOWN):
            log_error(f"Maximum drawdown reached: {drawdown:.2%} exceeds limit of {MAX_DRAWDOWN:.2%}")
            return False
            
        # Warning if approaching max drawdown
        if drawdown > abs(MAX_DRAWDOWN * 0.8):  # 80% of max drawdown
            print(f"⚠️ Warning: Approaching maximum drawdown: Current {drawdown:.2%}")
            
        return True
        
    except Exception as e:
        log_error(f"Error checking drawdown: {e}")
        return False  # Stop trading if there's an error

def get_position_info(symbol):
    try:
        positions = client.futures_position_information()
        for position in positions:
            if position['symbol'] == symbol:
                return {
                    'amount': float(position['positionAmt']),
                    'entry_price': float(position['entryPrice']),
                    'unrealized_pnl': float(position['unRealizedProfit']),
                    'leverage': float(position['leverage'])
                }
        return None
    except Exception as e:
        log_error(f"Error getting position info: {e}")
        return None

def run_bot():
    print("🚀 Binance Futures Trading Bot Started 🚀")
    
    initial_balance = get_balance()
    if initial_balance <= 0:
        log_error("Invalid initial balance. Cannot start trading.")
        return
        
    print(f"💰 Initial Balance: ${initial_balance:.2f} USDT")
    
    # Track highest balance for drawdown calculation
    highest_balance = initial_balance
    open_positions = {}  # Track open positions
    
    while True:
        try:
            current_balance = get_balance()
            
            # Update highest balance if current balance is higher
            if current_balance > highest_balance:
                highest_balance = current_balance
            
            # Calculate current drawdown from peak
            current_drawdown = (highest_balance - current_balance) / highest_balance
            
            print(f"\n💰 Current Balance: ${current_balance:.2f} USDT")
            print(f"📊 Peak Balance: ${highest_balance:.2f} USDT")
            print(f"📉 Current Drawdown: {current_drawdown:.2%}")
            
            # Check for maximum drawdown from peak
            if current_drawdown > abs(MAX_DRAWDOWN):
                log_error(f"Maximum drawdown reached: {current_drawdown:.2%} from peak of ${highest_balance:.2f}")
                break
                
            # Reduce position sizes if approaching max drawdown
            risk_multiplier = 1.0
            if current_drawdown > abs(MAX_DRAWDOWN * 0.7):  # At 70% of max drawdown
                risk_multiplier = 0.5  # Reduce risk by 50%
                print("⚠️ High drawdown detected - reducing position sizes by 50%")
            
            for symbol in FUTURES_SYMBOLS:
                print(f"\n🔍 Analyzing {symbol}...")
                
                # Get market data
                df = get_historical_data(symbol)
                if df is None:
                    continue
                    
                current_price = df['close'].iloc[-1]
                print(f"📈 Latest Price: ${current_price:.2f}")
                
                # Analyze market conditions
                market_conditions = analyze_market_conditions(df)
                print(f"📊 Market Conditions: Volatility={market_conditions['volatility']:.4f}, "
                      f"Volume Trend={'Up' if market_conditions['volume_trend'] else 'Down'}, "
                      f"Momentum={market_conditions['momentum']:.2%}")
                
                # Get position information
                position = get_position_info(symbol)
                
                # Check existing position
                if position and abs(float(position['amount'])) > 0:
                    print(f"Current Position: {position['amount']} contracts at ${position['entry_price']}")
                    print(f"Unrealized PnL: ${position['unrealized_pnl']:.2f}")
                    
                    # Check exit conditions
                    exit_trade, exit_reason = should_exit_trade(
                        position['entry_price'],
                        current_price,
                        "LONG" if position['amount'] > 0 else "SHORT"
                    )
                    
                    if exit_trade:
                        print(f"✅ Closing Trade: {symbol} | Reason: {exit_reason}")
                        close_side = SIDE_SELL if position['amount'] > 0 else SIDE_BUY
                        
                        order = place_order(
                            symbol=symbol,
                            side=close_side,
                            stop_loss=None,
                            take_profit=None
                        )
                        
                        if order:
                            log_trade_exit(symbol, "LONG" if position['amount'] > 0 else "SHORT",
                                         current_price, position['entry_price'], exit_reason)
                            if symbol in open_positions:
                                del open_positions[symbol]
                    
                    continue
                
                # Check entry conditions
                trade_signal, reason = should_enter_trade(df)
                print(f"📊 Signal: {trade_signal if trade_signal else 'HOLD'} | Reason: {reason}")
                
                if trade_signal and symbol not in open_positions:
                    print(f"🔹 Trade Decision: {trade_signal} for {symbol}")
                    
                    # Calculate stop loss and take profit levels
                    stop_loss = round(current_price * (1 - 0.02 if trade_signal == "LONG" else 1 + 0.02), 2)
                    take_profit = round(current_price * (1 + 0.04 if trade_signal == "LONG" else 1 - 0.04), 2)
                    
                    # Execute trade
                    order = place_order(
                        symbol=symbol,
                        side=SIDE_BUY if trade_signal == "LONG" else SIDE_SELL,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_multiplier=risk_multiplier
                    )
                    
                    if order:
                        log_trade(symbol, trade_signal, order['origQty'], current_price,
                                stop_loss, take_profit, "EXECUTED")
                        open_positions[symbol] = {
                            'side': trade_signal,
                            'entry_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'quantity': order['origQty']
                        }
            
            print("\nWaiting for next iteration...")
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n🛑 Bot Stopped Manually")
            break
        except Exception as e:
            log_error(str(e))
            print(f"❌ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_bot()
