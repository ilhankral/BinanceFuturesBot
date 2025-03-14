"""
Backtesting Module for Trading Bot
- Simulates trading strategies on historical data
- Calculates performance metrics
- Visualizes results
- Optimizes strategy parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Callable
import os
import json
from tqdm import tqdm

# Import custom modules
from src.config import (
    BINANCE_API_KEY, BINANCE_SECRET_KEY, FUTURES_SYMBOLS,
    TRADING_STRATEGIES, STRATEGY_PARAMS, DEFAULT_LEVERAGE,
    RISK_PER_TRADE, INITIAL_STOP_LOSS_PCT, INITIAL_TAKE_PROFIT_PCT,
    ENABLE_TRAILING_STOP, TRAILING_STOP_PERCENT
)

from src.strategies import (
    should_enter_trade, should_exit_trade, analyze_market_conditions,
    detect_market_regime
)

from src.indicators import (
    calculate_moving_averages, calculate_momentum_indicators,
    calculate_bollinger_bands, calculate_atr
)

from src.analytics import calculate_sharpe_ratio, calculate_max_drawdown
from src.logger import log_info, log_error
from binance.client import Client

# Initialize Binance client for historical data
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)


class BacktestResult:
    """Container for backtest results"""

    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.daily_returns = {}
        self.metrics = {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_return': 0,
            'average_return': 0,
            'average_win': 0,
            'average_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'avg_bars_in_trade': 0
        }
        self.parameters = {}
        self.symbol = ''
        self.start_date = None
        self.end_date = None

    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization"""
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns,
            'metrics': self.metrics,
            'parameters': self.parameters,
            'symbol': self.symbol,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None
        }

    def from_dict(self, data: Dict) -> 'BacktestResult':
        """Load result from dictionary"""
        self.trades = data.get('trades', [])
        self.equity_curve = data.get('equity_curve', [])
        self.daily_returns = data.get('daily_returns', {})
        self.metrics = data.get('metrics', {})
        self.parameters = data.get('parameters', {})
        self.symbol = data.get('symbol', '')

        if data.get('start_date'):
            self.start_date = datetime.fromisoformat(data['start_date'])
        if data.get('end_date'):
            self.end_date = datetime.fromisoformat(data['end_date'])

        return self

    def save(self, filename: str) -> bool:
        """Save result to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Convert to dictionary and save
            result_dict = self.to_dict()
            with open(filename, 'w') as f:
                json.dump(result_dict, f, indent=2)

            return True
        except Exception as e:
            log_error(f"Error saving backtest result: {e}")
            return False

    @classmethod
    def load(cls, filename: str) -> 'BacktestResult':
        """Load result from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            result = cls()
            return result.from_dict(data)
        except Exception as e:
            log_error(f"Error loading backtest result: {e}")
            return cls()

    def plot_equity_curve(self, filename: Optional[str] = None):
        """Plot equity curve"""
        try:
            if not self.equity_curve:
                log_error("No equity curve data to plot")
                return

            plt.figure(figsize=(12, 6))

            # Plot equity curve
            plt.plot(self.equity_curve, color='blue', linewidth=2)
            plt.title(f'Backtest Equity Curve - {self.symbol}')
            plt.xlabel('Bar Number')
            plt.ylabel('Account Value')
            plt.grid(True)

            # Calculate drawdown
            running_max = np.maximum.accumulate(self.equity_curve)
            drawdown = (self.equity_curve - running_max) / running_max

            # Create drawdown subplot
            plt.figure(figsize=(12, 6))
            plt.plot(drawdown, color='red', linewidth=1)
            plt.fill_between(np.arange(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
            plt.title(f'Drawdown - {self.symbol}')
            plt.xlabel('Bar Number')
            plt.ylabel('Drawdown %')
            plt.grid(True)

            # Display metrics
            metrics_text = f"""
            Total Trades: {self.metrics['total_trades']}
            Win Rate: {self.metrics['win_rate']:.2%}
            Profit Factor: {self.metrics['profit_factor']:.2f}
            Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}
            Max Drawdown: {self.metrics['max_drawdown']:.2%}
            Total Return: {self.metrics['total_return']:.2%}
            """
            plt.figtext(0.15, 0.15, metrics_text, fontsize=10, ha='left')

            # Save or display
            if filename:
                plt.savefig(filename)
                plt.close()
            else:
                plt.show()

        except Exception as e:
            log_error(f"Error plotting equity curve: {e}")

    def print_summary(self):
        """Print summary of backtest results"""
        print("\n" + "=" * 50)
        print(f"BACKTEST RESULTS - {self.symbol}")
        print("=" * 50)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Strategy Parameters: {self.parameters}")
        print("-" * 50)
        print(f"Total Trades: {self.metrics['total_trades']}")
        print(f"Win Rate: {self.metrics['win_rate']:.2%}")
        print(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {self.metrics['max_drawdown']:.2%}")
        print(f"Total Return: {self.metrics['total_return']:.2%}")
        print(f"Average Return per Trade: {self.metrics['average_return']:.2%}")
        print("-" * 50)
        print(f"Average Win: {self.metrics['average_win']:.2%}")
        print(f"Average Loss: {self.metrics['average_loss']:.2%}")
        print(f"Largest Win: {self.metrics['largest_win']:.2%}")
        print(f"Largest Loss: {self.metrics['largest_loss']:.2%}")
        print(f"Average Bars in Trade: {self.metrics['avg_bars_in_trade']:.1f}")
        print("=" * 50)

        # Print last few trades
        if self.trades:
            print("\nLast 5 Trades:")
            for trade in self.trades[-5:]:
                print(f"{trade['entry_time']} to {trade['exit_time']}: " +
                      f"{trade['side']} {trade['profit_pct']:.2%}")
        print("=" * 50)


def fetch_historical_data(symbol: str, interval: str = '1h',
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical data from Binance

    Args:
        symbol: Trading pair
        interval: Candlestick interval
        start_date: Start date for data
        end_date: End date for data
        limit: Maximum number of candles to fetch

    Returns:
        pd.DataFrame: Historical data
    """
    try:
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=30)

        # Convert dates to milliseconds
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        # Fetch klines from Binance
        klines = client.futures_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_ms,
            end_str=end_ms,
            limit=limit
        )

        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        # Convert types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])

        # Convert timestamps
        df['time'] = pd.to_datetime(df['time'], unit='ms')

        log_info(f"Fetched {len(df)} candles for {symbol} from {start_date} to {end_date}")
        return df

    except Exception as e:
        log_error(f"Error fetching historical data: {e}")
        return pd.DataFrame()


def run_backtest(symbol: str,
                 data: pd.DataFrame,
                 initial_capital: float = 10000.0,
                 fee_rate: float = 0.0004,  # 0.04% per trade
                 leverage: int = DEFAULT_LEVERAGE,
                 risk_per_trade: float = RISK_PER_TRADE,
                 stop_loss_pct: float = abs(INITIAL_STOP_LOSS_PCT),
                 take_profit_pct: float = INITIAL_TAKE_PROFIT_PCT,
                 enable_trailing: bool = ENABLE_TRAILING_STOP,
                 trailing_pct: float = TRAILING_STOP_PERCENT,
                 strategy_params: Optional[Dict] = None) -> BacktestResult:
    """
    Run backtest on historical data

    Args:
        symbol: Trading pair
        data: Historical data
        initial_capital: Starting capital
        fee_rate: Trading fee rate
        leverage: Trading leverage
        risk_per_trade: Risk per trade as percentage of capital
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        enable_trailing: Enable trailing stop
        trailing_pct: Trailing stop percentage
        strategy_params: Custom strategy parameters

    Returns:
        BacktestResult: Backtest results
    """
    try:
        # Initialize result object
        result = BacktestResult()
        result.symbol = symbol
        result.start_date = data['time'].iloc[0] if not data.empty else None
        result.end_date = data['time'].iloc[-1] if not data.empty else None

        # Store parameters
        result.parameters = {
            'leverage': leverage,
            'risk_per_trade': risk_per_trade,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'enable_trailing': enable_trailing,
            'trailing_pct': trailing_pct,
            'strategy_params': strategy_params or {}
        }

        # Prepare data - ensure we have all needed columns
        data = calculate_moving_averages(data)
        data = calculate_momentum_indicators(data)
        data = calculate_bollinger_bands(data)
        data = calculate_atr(data)

        if data.empty or len(data) < 100:
            log_error(f"Insufficient data for backtest: {len(data)} bars")
            return result

        # Initialize account and positions
        account = {
            'capital': initial_capital,
            'position': None,
            'entry_price': 0.0,
            'entry_time': None,
            'entry_bar': 0,
            'position_size': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'trailing_activated': False,
            'highest_price': 0.0,
            'lowest_price': float('inf')
        }

        equity_curve = [initial_capital]
        trades = []
        daily_returns = {}

        # Run through each bar (except the last few for indicator calculation)
        for i in range(200, len(data) - 1):
            # Current bar data
            current_bar = data.iloc[i].copy()
            next_bar = data.iloc[i + 1].copy()

            # Extract date for daily returns tracking
            bar_date = current_bar['time'].strftime('%Y-%m-%d')

            # Update highest/lowest prices if in position
            if account['position']:
                if account['position'] == 'LONG':
                    account['highest_price'] = max(account['highest_price'], current_bar['high'])
                else:  # SHORT
                    account['lowest_price'] = min(account['lowest_price'], current_bar['low'])

            # Check for exit if in position
            if account['position']:
                # Calculate current profit
                if account['position'] == 'LONG':
                    price_for_calculation = current_bar['close']
                    profit_pct = (price_for_calculation - account['entry_price']) / account['entry_price']
                else:  # SHORT
                    price_for_calculation = current_bar['close']
                    profit_pct = (account['entry_price'] - price_for_calculation) / account['entry_price']

                # Check stop loss (assume it happened at the low/high of the bar)
                stop_hit = False
                take_profit_hit = False

                if account['position'] == 'LONG':
                    if current_bar['low'] <= account['stop_loss']:
                        price_for_calculation = account['stop_loss']  # Use stop price
                        profit_pct = (price_for_calculation - account['entry_price']) / account['entry_price']
                        stop_hit = True
                    elif current_bar['high'] >= account['take_profit']:
                        price_for_calculation = account['take_profit']  # Use take profit price
                        profit_pct = (price_for_calculation - account['entry_price']) / account['entry_price']
                        take_profit_hit = True
                else:  # SHORT
                    if current_bar['high'] >= account['stop_loss']:
                        price_for_calculation = account['stop_loss']  # Use stop price
                        profit_pct = (account['entry_price'] - price_for_calculation) / account['entry_price']
                        stop_hit = True
                    elif current_bar['low'] <= account['take_profit']:
                        price_for_calculation = account['take_profit']  # Use take profit price
                        profit_pct = (account['entry_price'] - price_for_calculation) / account['entry_price']
                        take_profit_hit = True

                # Check for trailing stop adjustment
                if enable_trailing and not stop_hit and not take_profit_hit:
                    # Only activate trailing when profit exceeds threshold
                    if profit_pct >= 0.01 and not account['trailing_activated']:
                        account['trailing_activated'] = True

                    # Adjust stop loss if trailing is activated
                    if account['trailing_activated']:
                        if account['position'] == 'LONG':
                            new_stop = current_bar['close'] * (1 - trailing_pct)
                            if new_stop > account['stop_loss']:
                                account['stop_loss'] = new_stop
                        else:  # SHORT
                            new_stop = current_bar['close'] * (1 + trailing_pct)
                            if new_stop < account['stop_loss']:
                                account['stop_loss'] = new_stop

                # Check strategy exit signal
                exit_signal, exit_reason = should_exit_trade(
                    account['entry_price'],
                    current_bar['close'],
                    account['position'],
                    None,  # No market conditions for backtest
                    None,  # No market regime for backtest
                    profit_pct
                )

                # Execute exit if any exit condition is met
                if stop_hit or take_profit_hit or exit_signal:
                    # Calculate profit
                    profit_amount = account['position_size'] * profit_pct * leverage

                    # Apply trading fees
                    fees = account['position_size'] * fee_rate * 2  # Entry and exit fees
                    profit_amount -= fees

                    # Update account capital
                    account['capital'] += profit_amount

                    # Record trade
                    trade = {
                        'entry_time': account['entry_time'],
                        'exit_time': current_bar['time'],
                        'entry_price': account['entry_price'],
                        'exit_price': price_for_calculation,
                        'side': account['position'],
                        'profit_pct': profit_pct * 100,  # Convert to percentage
                        'profit_amount': profit_amount,
                        'bars_in_trade': i - account['entry_bar'],
                        'exit_reason': 'Stop Loss' if stop_hit else 'Take Profit' if take_profit_hit else exit_reason
                    }
                    trades.append(trade)

                    # Update daily returns
                    if bar_date in daily_returns:
                        daily_returns[bar_date] += profit_amount
                    else:
                        daily_returns[bar_date] = profit_amount

                    # Reset position
                    account['position'] = None
                    account['trailing_activated'] = False
                    account['highest_price'] = 0.0
                    account['lowest_price'] = float('inf')

            # Look for entry signals if not in position
            if not account['position']:
                # Create lookback window for strategy
                lookback_data = data.iloc[i - 200:i + 1].copy()

                # Get strategy signal
                signal, reason, score = should_enter_trade(
                    df_main=lookback_data,
                    market_conditions=analyze_market_conditions(lookback_data),
                    market_regime=detect_market_regime(lookback_data)
                )

                # Only take trades with score above threshold
                if signal and score >= 70:
                    # Calculate position size based on risk
                    account_risk = account['capital'] * risk_per_trade

                    if signal == 'LONG':
                        stop_price = current_bar['close'] * (1 - stop_loss_pct)
                        take_profit_price = current_bar['close'] * (1 + take_profit_pct)
                        risk_per_contract = (current_bar['close'] - stop_price) / current_bar['close']
                    else:  # SHORT
                        stop_price = current_bar['close'] * (1 + stop_loss_pct)
                        take_profit_price = current_bar['close'] * (1 - take_profit_pct)
                        risk_per_contract = (stop_price - current_bar['close']) / current_bar['close']

                    # Calculate position size
                    position_size = (account_risk / risk_per_contract) / leverage
                    position_size = min(position_size, account['capital'] * 0.95)  # Limit to 95% of capital

                    # Execute trade
                    entry_price = next_bar['open']  # Use next bar's open for entry

                    # Set up position
                    account['position'] = signal
                    account['entry_price'] = entry_price
                    account['entry_time'] = next_bar['time']
                    account['entry_bar'] = i + 1
                    account['position_size'] = position_size

                    # Set stop loss and take profit
                    if signal == 'LONG':
                        account['stop_loss'] = entry_price * (1 - stop_loss_pct)
                        account['take_profit'] = entry_price * (1 + take_profit_pct)
                        account['highest_price'] = entry_price
                    else:  # SHORT
                        account['stop_loss'] = entry_price * (1 + stop_loss_pct)
                        account['take_profit'] = entry_price * (1 - take_profit_pct)
                        account['lowest_price'] = entry_price

            # Update equity curve
            equity_curve.append(account['capital'])

        # If still in position at end of backtest, close it
        if account['position']:
            # Calculate final profit
            final_bar = data.iloc[-1]

            if account['position'] == 'LONG':
                profit_pct = (final_bar['close'] - account['entry_price']) / account['entry_price']
            else:  # SHORT
                profit_pct = (account['entry_price'] - final_bar['close']) / account['entry_price']

            # Calculate profit amount
            profit_amount = account['position_size'] * profit_pct * leverage

            # Apply trading fees
            fees = account['position_size'] * fee_rate * 2
            profit_amount -= fees

            # Update account capital
            account['capital'] += profit_amount

            # Record trade
            trade = {
                'entry_time': account['entry_time'],
                'exit_time': final_bar['time'],
                'entry_price': account['entry_price'],
                'exit_price': final_bar['close'],
                'side': account['position'],
                'profit_pct': profit_pct * 100,
                'profit_amount': profit_amount,
                'bars_in_trade': len(data) - 1 - account['entry_bar'],
                'exit_reason': 'End of Backtest'
            }
            trades.append(trade)

            # Update final equity
            equity_curve[-1] = account['capital']

        # Store results
        result.trades = trades
        result.equity_curve = equity_curve
        result.daily_returns = daily_returns

        # Calculate performance metrics
        result.metrics['total_trades'] = len(trades)

        if trades:
            # Win rate
            winning_trades = [t for t in trades if t['profit_pct'] > 0]
            result.metrics['win_rate'] = len(winning_trades) / len(trades)

            # Profit factor
            gross_profit = sum(t['profit_amount'] for t in trades if t['profit_amount'] > 0)
            gross_loss = abs(sum(t['profit_amount'] for t in trades if t['profit_amount'] <= 0))
            result.metrics['profit_factor'] = gross_profit / gross_loss if gross_loss else float('inf')

            # Returns
            result.metrics['total_return'] = (account['capital'] - initial_capital) / initial_capital
            result.metrics['average_return'] = result.metrics['total_return'] / len(trades)

            # Average trade stats
            win_pcts = [t['profit_pct'] for t in trades if t['profit_pct'] > 0]
            loss_pcts = [t['profit_pct'] for t in trades if t['profit_pct'] <= 0]

            result.metrics['average_win'] = sum(win_pcts) / len(win_pcts) if win_pcts else 0
            result.metrics['average_loss'] = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0
            result.metrics['largest_win'] = max(win_pcts) if win_pcts else 0
            result.metrics['largest_loss'] = min(loss_pcts) if loss_pcts else 0

            # Average bars in trade
            result.metrics['avg_bars_in_trade'] = sum(t['bars_in_trade'] for t in trades) / len(trades)

        # Calculate risk metrics
        daily_returns_list = list(daily_returns.values())
        if daily_returns_list:
            result.metrics['sharpe_ratio'] = calculate_sharpe_ratio(daily_returns_list)

        # Max drawdown
        result.metrics['max_drawdown'] = calculate_max_drawdown(equity_curve)

        return result

    except Exception as e:
        log_error(f"Error in backtest: {e}")
        import traceback
        traceback.print_exc()
        return BacktestResult()


def optimize_parameters(symbol: str,
                        data: pd.DataFrame,
                        param_grid: Dict[str, List],
                        metric: str = 'sharpe_ratio',
                        top_n: int = 3) -> List[Tuple[Dict, BacktestResult]]:
    """
    Optimize strategy parameters using grid search

    Args:
        symbol: Trading pair
        data: Historical data
        param_grid: Dictionary mapping parameter names to lists of values
        metric: Metric to optimize (e.g., 'sharpe_ratio', 'total_return')
        top_n: Number of top parameter sets to return

    Returns:
        List[Tuple[Dict, BacktestResult]]: Top parameter sets and results
    """
    try:
        # Generate all parameter combinations
        import itertools

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        log_info(f"Running optimization with {len(param_combinations)} parameter combinations")

        results = []

        # Run backtest for each parameter combination
        for params in tqdm(param_combinations, desc="Optimizing"):
            # Create parameter dictionary
            param_dict = {name: value for name, value in zip(param_names, params)}

            # Extract common parameters
            leverage = param_dict.pop('leverage', DEFAULT_LEVERAGE)
            risk_per_trade = param_dict.pop('risk_per_trade', RISK_PER_TRADE)
            stop_loss_pct = param_dict.pop('stop_loss_pct', abs(INITIAL_STOP_LOSS_PCT))
            take_profit_pct = param_dict.pop('take_profit_pct', INITIAL_TAKE_PROFIT_PCT)
            enable_trailing = param_dict.pop('enable_trailing', ENABLE_TRAILING_STOP)
            trailing_pct = param_dict.pop('trailing_pct', TRAILING_STOP_PERCENT)

            # Run backtest with these parameters
            result = run_backtest(
                symbol=symbol,
                data=data,
                leverage=leverage,
                risk_per_trade=risk_per_trade,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                enable_trailing=enable_trailing,
                trailing_pct=trailing_pct,
                strategy_params=param_dict
            )

            # Add to results if valid
            if result.metrics['total_trades'] > 0:
                results.append((param_dict, result))

        # Sort by chosen metric
        def get_metric(result_tuple):
            # For some metrics, higher is better
            if metric in ['sharpe_ratio', 'profit_factor', 'win_rate', 'total_return']:
                return -result_tuple[1].metrics.get(metric, 0)  # Negative for descending order
            # For others, lower is better
            else:
                return result_tuple[1].metrics.get(metric, float('inf'))

        sorted_results = sorted(results, key=get_metric)

        # Return top N results
        return sorted_results[:top_n]

    except Exception as e:
        log_error(f"Error in parameter optimization: {e}")
        return []


def run_backtest_for_symbols(symbols: List[str],
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             save_results: bool = True,
                             output_dir: str = 'backtest_results') -> Dict[str, BacktestResult]:
    """
    Run backtest for multiple symbols

    Args:
        symbols: List of trading pairs
        start_date: Start date for data
        end_date: End date for data
        save_results: Whether to save results to files
        output_dir: Directory to save results

    Returns:
        Dict[str, BacktestResult]: Backtest results by symbol
    """
    if not symbols:
        log_error("No symbols provided for backtest")
        return {}

    try:
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=90)

        results = {}

        # Run backtest for each symbol
        for symbol in symbols:
            log_info(f"Running backtest for {symbol}")

            # Fetch historical data
            data = fetch_historical_data(
                symbol=symbol,
                interval='1h',
                start_date=start_date,
                end_date=end_date,
                limit=1000
            )

            if data.empty:
                log_error(f"No data available for {symbol}")
                continue

            # Run backtest
            result = run_backtest(symbol=symbol, data=data)

            # Store result
            results[symbol] = result

            # Save result if requested
            if save_results:
                # Create output directory
                os.makedirs(output_dir, exist_ok=True)

                # Generate filename
                filename = f"{output_dir}/{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"

                # Save result
                result.save(filename)

                # Save equity curve plot
                plot_filename = f"{output_dir}/{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
                result.plot_equity_curve(plot_filename)

            # Print summary
            result.print_summary()

        return results

    except Exception as e:
        log_error(f"Error running backtest for symbols: {e}")
        return {}


def compare_strategies(symbols: List[str], strategies: List[str],
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Compare performance of different strategies

    Args:
        symbols: List of trading pairs
        strategies: List of strategy names
        start_date: Start date for data
        end_date: End date for data

    Returns:
        pd.DataFrame: Comparison of strategy performance
    """
    try:
        # Set default dates
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=90)

        # Prepare results dataframe
        columns = ['Symbol', 'Strategy', 'Total Trades', 'Win Rate', 'Profit Factor',
                   'Sharpe Ratio', 'Max Drawdown', 'Total Return']
        results_data = []

        # Run backtest for each symbol and strategy
        for symbol in symbols:
            log_info(f"Comparing strategies for {symbol}")

            # Fetch historical data
            data = fetch_historical_data(
                symbol=symbol,
                interval='1h',
                start_date=start_date,
                end_date=end_date,
                limit=1000
            )

            if data.empty:
                log_error(f"No data available for {symbol}")
                continue

            # Test each strategy
            for strategy in strategies:
                log_info(f"Testing {strategy} on {symbol}")

                # Run backtest with specified strategy
                strategy_params = {'strategies': [strategy]}

                result = run_backtest(
                    symbol=symbol,
                    data=data,
                    strategy_params=strategy_params
                )

                # Add to results
                if result.metrics['total_trades'] > 0:
                    results_data.append([
                        symbol,
                        strategy,
                        result.metrics['total_trades'],
                        result.metrics['win_rate'],
                        result.metrics['profit_factor'],
                        result.metrics['sharpe_ratio'],
                        result.metrics['max_drawdown'],
                        result.metrics['total_return']
                    ])

        # Create DataFrame
        results_df = pd.DataFrame(results_data, columns=columns)

        return results_df

    except Exception as e:
        log_error(f"Error comparing strategies: {e}")
        return pd.DataFrame()


def main():
    """Main function to run backtest from command line"""
    import argparse

    parser = argparse.ArgumentParser(description="Backtest trading strategies")
    parser.add_argument("--symbols", type=str, default="BTCUSDT", help="Comma-separated list of symbols")
    parser.add_argument("--days", type=int, default=90, help="Number of days to backtest")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    parser.add_argument("--compare", action="store_true", help="Compare strategies")
    parser.add_argument("--output-dir", type=str, default="backtest_results", help="Output directory")

    args = parser.parse_args()

    # Parse symbols
    symbols = args.symbols.split(',')

    # Set dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    if args.optimize:
        # Run parameter optimization
        for symbol in symbols:
            data = fetch_historical_data(
                symbol=symbol,
                interval='1h',
                start_date=start_date,
                end_date=end_date,
                limit=1000
            )

            param_grid = {
                'leverage': [3, 5, 10],
                'risk_per_trade': [0.01, 0.02, 0.03],
                'stop_loss_pct': [0.01, 0.02, 0.03],
                'take_profit_pct': [0.02, 0.04, 0.06],
                'ema_fast': [20, 50, 100],
                'ema_slow': [100, 200]
            }

            results = optimize_parameters(
                symbol=symbol,
                data=data,
                param_grid=param_grid,
                metric='sharpe_ratio',
                top_n=3
            )

            print(f"Top parameters for {symbol}:")
            for i, (params, result) in enumerate(results):
                print(f"Rank {i + 1}:")
                print(f"Parameters: {params}")
                result.print_summary()

    elif args.compare:
        # Compare strategies
        strategies = TRADING_STRATEGIES

        results_df = compare_strategies(
            symbols=symbols,
            strategies=strategies,
            start_date=start_date,
            end_date=end_date
        )

        print("Strategy Comparison:")
        print(results_df)

        # Save to CSV
        results_df.to_csv(f"{args.output_dir}/strategy_comparison.csv", index=False)

    else:
        # Run standard backtest
        results = run_backtest_for_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            save_results=True,
            output_dir=args.output_dir
        )

        # Print overall summary
        print("\nOverall Summary:")
        for symbol, result in results.items():
            print(f"{symbol}: Trades={result.metrics['total_trades']}, " +
                  f"Win Rate={result.metrics['win_rate']:.2%}, " +
                  f"Return={result.metrics['total_return']:.2%}")


if __name__ == "__main__":
    main()