"""
Performance Analytics Module
- Trade performance tracking and analysis
- Risk metrics calculation (Sharpe, drawdown, win rate)
- Performance visualization
- Notifications for important events
- Strategy performance evaluation
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import requests
from scipy import stats

# Import configuration
from src.config import (
    PERFORMANCE_DATA_FILE, ENABLE_PERFORMANCE_ANALYTICS,
    ENABLE_NOTIFICATIONS, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    DISCORD_WEBHOOK_URL
)

# Import logger
from src.logger import log_info, log_error

# Global store for performance data
performance_data = {
    'trades': [],  # List of all trades
    'daily_returns': {},  # Daily P&L
    'equity_curve': {},  # Account equity over time
    'strategy_performance': {},  # Performance by strategy
    'pair_performance': {},  # Performance by trading pair
    'max_drawdown': 0,  # Maximum drawdown percentage
    'current_drawdown': 0,  # Current drawdown from peak
    'win_rate': 0,  # Percentage of winning trades
    'profit_factor': 0,  # Gross profit / gross loss
    'sharpe_ratio': 0,  # Sharpe ratio
    'sortino_ratio': 0,  # Sortino ratio (downside risk adjusted)
    'avg_win': 0,  # Average winning trade (%)
    'avg_loss': 0,  # Average losing trade (%)
    'largest_win': 0,  # Largest winning trade (%)
    'largest_loss': 0,  # Largest losing trade (%)
    'avg_trade_duration': 0,  # Average holding time
    'total_fees': 0,  # Total trading fees paid
    'last_updated': None  # Last update timestamp
}


def initialize_analytics():
    """Initialize analytics by loading saved data if available"""
    global performance_data

    if not ENABLE_PERFORMANCE_ANALYTICS:
        return

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(PERFORMANCE_DATA_FILE), exist_ok=True)

        # Load existing data if available
        if os.path.exists(PERFORMANCE_DATA_FILE):
            with open(PERFORMANCE_DATA_FILE, 'r') as f:
                loaded_data = json.load(f)

                # Convert string dates back to datetime where needed
                if 'trades' in loaded_data:
                    for trade in loaded_data['trades']:
                        if 'entry_time' in trade and isinstance(trade['entry_time'], str):
                            trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])
                        if 'exit_time' in trade and isinstance(trade['exit_time'], str):
                            trade['exit_time'] = datetime.fromisoformat(trade['exit_time'])

                # Update performance data
                performance_data.update(loaded_data)

                log_info(f"Loaded performance data: {len(performance_data['trades'])} trades")
        else:
            log_info("No existing performance data found, starting fresh")

    except Exception as e:
        log_error(f"Error initializing analytics: {e}")


def save_performance_data():
    """Save performance data to file"""
    if not ENABLE_PERFORMANCE_ANALYTICS:
        return

    try:
        # Prepare data for serialization by converting datetime objects
        data_copy = performance_data.copy()

        if 'trades' in data_copy:
            for trade in data_copy['trades']:
                if 'entry_time' in trade and isinstance(trade['entry_time'], datetime):
                    trade['entry_time'] = trade['entry_time'].isoformat()
                if 'exit_time' in trade and isinstance(trade['exit_time'], datetime):
                    trade['exit_time'] = trade['exit_time'].isoformat()
                # Convert timedelta objects to seconds
                if 'duration' in trade and isinstance(trade['duration'], timedelta):
                    trade['duration_seconds'] = trade['duration'].total_seconds()
                    del trade['duration']

        # Update last updated timestamp
        data_copy['last_updated'] = datetime.now().isoformat()

        # Save to file
        with open(PERFORMANCE_DATA_FILE, 'w') as f:
            json.dump(data_copy, f, indent=2)

        log_info(f"Performance data saved to {PERFORMANCE_DATA_FILE}")

    except Exception as e:
        log_error(f"Error saving performance data: {e}")


def update_trade_performance(trade_result: Dict):
    """
    Update performance metrics with a new trade result

    Args:
        trade_result: Dictionary with trade details
    """
    if not ENABLE_PERFORMANCE_ANALYTICS:
        return

    try:
        # Add trade to history
        performance_data['trades'].append(trade_result)

        # Update metrics
        _update_performance_metrics()

        # Save updated data
        save_performance_data()

        # Notify if significant trade
        profit_pct = trade_result.get('profit_pct', 0)
        if abs(profit_pct) > 3.0:  # Notify for trades with >3% profit or loss
            notify_important_event(
                f"📊 Trade closed: {trade_result['symbol']} {trade_result['side']} " +
                f"with {profit_pct:.2f}% profit"
            )

    except Exception as e:
        log_error(f"Error updating trade performance: {e}")


def _update_performance_metrics():
    """Recalculate all performance metrics based on trade history"""
    global performance_data

    try:
        trades = performance_data['trades']
        if not trades:
            return

        # Calculate win rate
        winning_trades = [t for t in trades if t.get('profit_pct', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit_pct', 0) <= 0]

        performance_data['win_rate'] = len(winning_trades) / len(trades) if trades else 0

        # Calculate average profits
        if winning_trades:
            performance_data['avg_win'] = sum(t.get('profit_pct', 0) for t in winning_trades) / len(winning_trades)
            performance_data['largest_win'] = max(t.get('profit_pct', 0) for t in winning_trades)

        if losing_trades:
            performance_data['avg_loss'] = sum(t.get('profit_pct', 0) for t in losing_trades) / len(losing_trades)
            performance_data['largest_loss'] = min(t.get('profit_pct', 0) for t in losing_trades)

        # Calculate profit factor
        gross_profit = sum(t.get('profit_amount', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('profit_amount', 0) for t in losing_trades))

        performance_data['profit_factor'] = gross_profit / gross_loss if gross_loss else float('inf')

        # Calculate daily returns
        daily_returns = {}

        for trade in trades:
            exit_time = trade.get('exit_time')
            if not exit_time:
                continue

            date_key = exit_time.strftime('%Y-%m-%d')
            profit = trade.get('profit_amount', 0)

            if date_key in daily_returns:
                daily_returns[date_key] += profit
            else:
                daily_returns[date_key] = profit

        performance_data['daily_returns'] = daily_returns

        # Calculate trade durations
        durations = []
        for trade in trades:
            if 'duration' in trade and isinstance(trade['duration'], timedelta):
                durations.append(trade['duration'].total_seconds())
            elif 'duration_seconds' in trade:
                durations.append(trade['duration_seconds'])

        if durations:
            avg_duration_seconds = sum(durations) / len(durations)
            performance_data['avg_trade_duration'] = avg_duration_seconds

        # Update performance by strategy
        strategy_performance = {}

        for trade in trades:
            strategy = trade.get('strategy', 'Unknown')

            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    'count': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit_sum': 0,
                    'win_rate': 0,
                    'avg_profit': 0
                }

            strategy_performance[strategy]['count'] += 1

            if trade.get('profit_pct', 0) > 0:
                strategy_performance[strategy]['wins'] += 1
            else:
                strategy_performance[strategy]['losses'] += 1

            strategy_performance[strategy]['profit_sum'] += trade.get('profit_pct', 0)

        # Calculate strategy metrics
        for strategy, data in strategy_performance.items():
            if data['count'] > 0:
                data['win_rate'] = data['wins'] / data['count']
                data['avg_profit'] = data['profit_sum'] / data['count']

        performance_data['strategy_performance'] = strategy_performance

        # Update performance by trading pair
        pair_performance = {}

        for trade in trades:
            symbol = trade.get('symbol', 'Unknown')

            if symbol not in pair_performance:
                pair_performance[symbol] = {
                    'count': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit_sum': 0,
                    'win_rate': 0,
                    'avg_profit': 0
                }

            pair_performance[symbol]['count'] += 1

            if trade.get('profit_pct', 0) > 0:
                pair_performance[symbol]['wins'] += 1
            else:
                pair_performance[symbol]['losses'] += 1

            pair_performance[symbol]['profit_sum'] += trade.get('profit_pct', 0)

        # Calculate pair metrics
        for symbol, data in pair_performance.items():
            if data['count'] > 0:
                data['win_rate'] = data['wins'] / data['count']
                data['avg_profit'] = data['profit_sum'] / data['count']

        performance_data['pair_performance'] = pair_performance

        # Calculate risk-adjusted metrics if we have enough data
        if len(daily_returns) > 5:
            returns_series = pd.Series(list(daily_returns.values()))

            # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
            sharpe = returns_series.mean() / returns_series.std() if returns_series.std() != 0 else 0
            performance_data['sharpe_ratio'] = sharpe * (252 ** 0.5)  # Annualized

            # Sortino ratio (downside risk only)
            negative_returns = returns_series[returns_series < 0]
            downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0.0001
            sortino = returns_series.mean() / downside_deviation if downside_deviation != 0 else 0
            performance_data['sortino_ratio'] = sortino * (252 ** 0.5)  # Annualized

    except Exception as e:
        log_error(f"Error updating performance metrics: {e}")


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for a series of returns

    Args:
        returns: List of periodic returns
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        float: Annualized Sharpe ratio
    """
    if not returns:
        return 0.0

    try:
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate

        if len(returns) <= 1 or np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        annualized_sharpe = sharpe * np.sqrt(252)  # Assuming daily returns, annualize

        return annualized_sharpe

    except Exception as e:
        log_error(f"Error calculating Sharpe ratio: {e}")
        return 0.0


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate maximum drawdown from equity curve

    Args:
        equity_curve: List of account equity values over time

    Returns:
        float: Maximum drawdown as a percentage
    """
    if not equity_curve:
        return 0.0

    try:
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(min(drawdown))

        return max_drawdown

    except Exception as e:
        log_error(f"Error calculating max drawdown: {e}")
        return 0.0


def log_daily_statistics(trade_history: List[Dict]):
    """
    Log daily performance statistics

    Args:
        trade_history: List of trades
    """
    if not ENABLE_PERFORMANCE_ANALYTICS or not trade_history:
        return

    try:
        # Filter trades from today
        today = datetime.now().date()
        today_trades = [t for t in trade_history if
                        (t.get('exit_time') and t['exit_time'].date() == today)]

        if not today_trades:
            return

        # Calculate today's performance
        today_profit = sum(t.get('profit_amount', 0) for t in today_trades)
        win_trades = [t for t in today_trades if t.get('profit_pct', 0) > 0]
        loss_trades = [t for t in today_trades if t.get('profit_pct', 0) <= 0]

        win_rate = len(win_trades) / len(today_trades) if today_trades else 0

        # Get best and worst trade
        best_trade = max(today_trades, key=lambda x: x.get('profit_pct', 0)) if today_trades else None
        worst_trade = min(today_trades, key=lambda x: x.get('profit_pct', 0)) if today_trades else None

        # Log statistics
        log_message = (
                f"📊 Daily Statistics ({today})\n" +
                f"Trades: {len(today_trades)}, Win Rate: {win_rate:.1%}\n" +
                f"P&L: ${today_profit:.2f}\n"
        )

        if best_trade:
            log_message += (
                    f"Best Trade: {best_trade['symbol']} {best_trade['side']} " +
                    f"{best_trade.get('profit_pct', 0):.2f}%\n"
            )

        if worst_trade:
            log_message += (
                    f"Worst Trade: {worst_trade['symbol']} {worst_trade['side']} " +
                    f"{worst_trade.get('profit_pct', 0):.2f}%\n"
            )

        log_info(log_message)

        # Notify daily summary
        notify_important_event(log_message)

    except Exception as e:
        log_error(f"Error logging daily statistics: {e}")


def generate_performance_report() -> Dict:
    """
    Generate a comprehensive performance report

    Returns:
        Dict: Performance metrics and statistics
    """
    if not ENABLE_PERFORMANCE_ANALYTICS:
        return {}

    try:
        # Start with existing performance data
        report = {
            'summary': {
                'total_trades': len(performance_data['trades']),
                'win_rate': performance_data['win_rate'],
                'profit_factor': performance_data['profit_factor'],
                'sharpe_ratio': performance_data['sharpe_ratio'],
                'sortino_ratio': performance_data['sortino_ratio'],
                'max_drawdown': performance_data['max_drawdown'],
                'avg_trade_duration': performance_data['avg_trade_duration'],
                'avg_win': performance_data['avg_win'],
                'avg_loss': performance_data['avg_loss']
            },
            'by_strategy': performance_data['strategy_performance'],
            'by_pair': performance_data['pair_performance'],
            'recent_trades': performance_data['trades'][-10:] if performance_data['trades'] else []
        }

        # Calculate monthly returns
        monthly_returns = {}

        for date_str, profit in performance_data['daily_returns'].items():
            date = datetime.strptime(date_str, '%Y-%m-%d')
            month_key = date.strftime('%Y-%m')

            if month_key in monthly_returns:
                monthly_returns[month_key] += profit
            else:
                monthly_returns[month_key] = profit

        report['monthly_returns'] = monthly_returns

        # Calculate metrics by time of day
        trades_by_hour = {}

        for trade in performance_data['trades']:
            if 'entry_time' not in trade:
                continue

            hour = trade['entry_time'].hour
            hour_key = f"{hour:02d}:00"

            if hour_key not in trades_by_hour:
                trades_by_hour[hour_key] = {
                    'count': 0,
                    'wins': 0,
                    'profit_sum': 0
                }

            trades_by_hour[hour_key]['count'] += 1

            if trade.get('profit_pct', 0) > 0:
                trades_by_hour[hour_key]['wins'] += 1

            trades_by_hour[hour_key]['profit_sum'] += trade.get('profit_pct', 0)

        # Calculate hourly metrics
        for hour, data in trades_by_hour.items():
            if data['count'] > 0:
                data['win_rate'] = data['wins'] / data['count']
                data['avg_profit'] = data['profit_sum'] / data['count']

        report['by_hour'] = trades_by_hour

        return report

    except Exception as e:
        log_error(f"Error generating performance report: {e}")
        return {}


def notify_important_event(message: str):
    """
    Send notification for important events

    Args:
        message: Notification message
    """
    if not ENABLE_NOTIFICATIONS:
        return

    try:
        # Telegram notifications
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            _send_telegram_notification(message)

        # Discord notifications
        if DISCORD_WEBHOOK_URL:
            _send_discord_notification(message)

    except Exception as e:
        log_error(f"Error sending notification: {e}")


def _send_telegram_notification(message: str):
    """Send a message via Telegram Bot API"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

    except Exception as e:
        log_error(f"Error sending Telegram notification: {e}")


def _send_discord_notification(message: str):
    """Send a message via Discord webhook"""
    if not DISCORD_WEBHOOK_URL:
        return

    try:
        # Format message for Discord
        payload = {
            'content': message,
            'username': 'Trading Bot'
        }

        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        response.raise_for_status()

    except Exception as e:
        log_error(f"Error sending Discord notification: {e}")


def plot_equity_curve(filepath: str = 'equity_curve.png'):
    """
    Generate and save an equity curve plot

    Args:
        filepath: Where to save the plot
    """
    if not ENABLE_PERFORMANCE_ANALYTICS:
        return

    try:
        # Check if we have enough data
        if not performance_data['daily_returns']:
            log_info("Not enough data to plot equity curve")
            return

        # Convert daily returns to a DataFrame
        daily_returns_df = pd.DataFrame(
            [(datetime.strptime(date, '%Y-%m-%d'), profit)
             for date, profit in performance_data['daily_returns'].items()],
            columns=['date', 'profit']
        )

        # Sort by date
        daily_returns_df.sort_values('date', inplace=True)

        # Calculate cumulative returns
        daily_returns_df['cumulative_profit'] = daily_returns_df['profit'].cumsum()

        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(daily_returns_df['date'], daily_returns_df['cumulative_profit'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Profit (USDT)')
        plt.grid(True)

        # Add drawdown
        running_max = daily_returns_df['cumulative_profit'].cummax()
        drawdown = (daily_returns_df['cumulative_profit'] - running_max)

        # Plot drawdown as a filled area
        plt.fill_between(
            daily_returns_df['date'],
            drawdown,
            0,
            where=(drawdown < 0),
            color='red',
            alpha=0.3
        )

        # Save plot
        plt.savefig(filepath)
        log_info(f"Equity curve saved to {filepath}")

    except Exception as e:
        log_error(f"Error plotting equity curve: {e}")


def calculate_optimal_leverage(max_drawdown: float, target_risk: float = 0.3) -> int:
    """
    Calculate optimal leverage based on historical performance

    Args:
        max_drawdown: Maximum historical drawdown (as a decimal)
        target_risk: Target risk (as a decimal)

    Returns:
        int: Recommended leverage
    """
    try:
        if max_drawdown <= 0:
            return 1  # Default to 1x if no drawdown data

        # Calculate leverage that would result in max drawdown of target_risk
        optimal = target_risk / max_drawdown

        # Round down to be conservative
        return max(1, min(125, int(optimal)))

    except Exception as e:
        log_error(f"Error calculating optimal leverage: {e}")
        return 1


def analyze_strategy_correlations() -> pd.DataFrame:
    """
    Analyze correlations between different trading strategies

    Returns:
        pd.DataFrame: Correlation matrix
    """
    if not ENABLE_PERFORMANCE_ANALYTICS or not performance_data['trades']:
        return pd.DataFrame()

    try:
        # Create a DataFrame with strategy returns
        strategy_returns = {}

        for trade in performance_data['trades']:
            strategy = trade.get('strategy', 'Unknown')
            exit_time = trade.get('exit_time')

            if not exit_time:
                continue

            date_key = exit_time.strftime('%Y-%m-%d')
            profit = trade.get('profit_pct', 0)

            if strategy not in strategy_returns:
                strategy_returns[strategy] = {}

            if date_key in strategy_returns[strategy]:
                strategy_returns[strategy][date_key] += profit
            else:
                strategy_returns[strategy][date_key] = profit

        # Convert to DataFrame
        strategies = list(strategy_returns.keys())
        all_dates = set()

        for strategy_data in strategy_returns.values():
            all_dates.update(strategy_data.keys())

        # Create DataFrame with all dates and strategies
        correlation_data = []

        for date in sorted(all_dates):
            row = {'date': date}

            for strategy in strategies:
                row[strategy] = strategy_returns[strategy].get(date, 0)

            correlation_data.append(row)

        # Convert to DataFrame and calculate correlation
        if correlation_data:
            df = pd.DataFrame(correlation_data)
            df.set_index('date', inplace=True)

            correlation_matrix = df.corr()
            return correlation_matrix

        return pd.DataFrame()

    except Exception as e:
        log_error(f"Error analyzing strategy correlations: {e}")
        return pd.DataFrame()


# Initialize analytics on module import
initialize_analytics()