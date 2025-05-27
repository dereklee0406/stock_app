"""
Strategy module for Stock Analysis Tool.
Implements and backtests trading strategies.
"""
import os
import json
import pandas as pd
import numpy as np
from pprint import pformat
from logs.logger_setup import LoggerSetup
from config.config_loader import ConfigLoader

def save_trade_log_and_summary(trade_log_df: pd.DataFrame, results: dict, symbol: str):
    """Save trade log and summary results as CSV and JSON with robust error handling."""
    os.makedirs('backtesting_results', exist_ok=True)
    try:
        trade_log_df.to_csv(f'backtesting_results/{symbol}_trade_log.csv', index=False)
    except Exception as e:
        print(f"Error saving trade log CSV: {e}")
    try:
        # Convert all non-JSON-serializable types to string or float
        def make_json_serializable(val):
            import numpy as np
            import pandas as pd
            if isinstance(val, (pd.Timestamp, pd.Timedelta)):
                return str(val)
            if isinstance(val, (np.generic,)):
                return float(val)
            return val
        # Use DataFrame.astype(object) and .map for elementwise mapping
        trade_log_json = trade_log_df.astype(object).map(make_json_serializable).to_dict(orient='records')
        with open(f'backtesting_results/{symbol}_trade_log.json', 'w', encoding='utf-8') as f:
            json.dump(trade_log_json, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving trade log JSON: {e}")
    try:
        with open(f'backtesting_results/{symbol}_backtest_summary.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving summary JSON: {e}")

class Strategy:
    """
    Develops, integrates, and backtests trading strategies.
    """
    def __init__(self, config: dict = None, log_level: str = 'INFO'):
        self.config = config or {}
        self.logger = LoggerSetup(log_level=log_level).get_logger()

    def _dynamic_stop_loss(self, row, price):
        """Calculate dynamic stop-loss percentage based on ATR or config."""
        if 'atr_14' in row:
            return max(self.config.get('stop_loss_pct', 0.05), row['atr_14'] / price)
        return self.config.get('stop_loss_pct', 0.05)

    def _dynamic_take_profit(self, row, entry_price):
        """Calculate dynamic take-profit threshold based on ATR or config."""
        if 'atr_14' in row and entry_price > 0:
            return 2 * row['atr_14'] / entry_price
        return 0.20

    def _allow_entry(self, signal, confidence, position):
        """Require multi-indicator confirmation for entry (signal must be 'buy' and confidence > threshold)."""
        confidence_threshold = int(self.config.get('confidence_threshold', 75))
        return signal == 'buy' and confidence > confidence_threshold and position == 0

    def _should_reenter(self, last_signal, current_signal, confidence, position):
        """Only allow re-entry after a trade if a new signal is present and confidence is high."""
        confidence_threshold = int(self.config.get('confidence_threshold', 75))
        return current_signal == 'buy' and confidence > confidence_threshold and position == 0 and last_signal != 'buy'

    def _apply_cooldown(self, cooldown_counter, position):
        """Reduce overtrading by enforcing a cooldown period after a loss."""
        if cooldown_counter > 0 and position == 0:
            return True
        return False

    def _safe_compare(self, a, b, op):
        """Safely compare two values, handling None as always False."""
        if a is None or b is None:
            return False
        return op(a, b)

    def backtest(self, df: pd.DataFrame, signals: pd.DataFrame, symbol: str = None) -> dict:
        """
        Backtests a strategy and returns performance metrics.
        Saves results and equity curve as CSV/PNG in backtesting_results/.

        Configurable parameters (via config or .env):
        - confidence_threshold: int, minimum confidence for entry/re-entry
        - cooldown_period: int, days to wait after loss or rapid trades
        - stop_loss_pct: float, fallback stop-loss percent
        - min_holding_period: int, minimum days to hold a position
        - slippage: float, per-trade slippage
        - fee: float, per-trade fee
        - take_profit_atr: float, ATR multiplier for take-profit
        - consecutive_loss_threshold: int, number of consecutive losses to trigger cooldown (default 3)
        - rapid_trade_interval: int, max days between trades to trigger cooldown (default 2)
        """
        if df.empty or signals.empty:
            self.logger.error("No data or signals for backtesting.")
            return {}
        try:
            # Read start date and risk-free rate from .env configuration
            config_loader = ConfigLoader()
            start_date = config_loader.get('START_DATE', '2025-01-01')
            risk_free_rate = config_loader.get_float('RISK_FREE_RATE', 0.05)  # annualized, default 5%
            daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1  # convert to daily compounding
            start_of_period = pd.to_datetime(start_date)
            df = df.copy()
            df = df[df.index >= start_of_period]
            signals = signals[signals.index >= start_of_period]
            # Ensure both dataframes are aligned to the same start date
            common_index = df.index.intersection(signals.index)
            df = df.loc[common_index]
            signals = signals.loc[common_index]
            df['signal'] = signals['signal']
            df['returns'] = df['close'].pct_change().fillna(0)
            
            confidence_threshold = int(config_loader.get('CONFIDENCE_THRESHOLD', self.config.get('confidence_threshold', 85)))  # was 75
            cooldown_period = int(config_loader.get('COOLDOWN_PERIOD', self.config.get('cooldown_period', 5)))
            stop_loss_pct = float(config_loader.get('STOP_LOSS_PCT', self.config.get('stop_loss_pct', 0.05)))
            min_holding_period = int(config_loader.get('MIN_HOLDING_PERIOD', self.config.get('min_holding_period', 10)))  # was 5
            slippage = float(config_loader.get('SLIPPAGE', self.config.get('slippage', 0.0005)))
            fee = float(config_loader.get('FEE', self.config.get('fee', 0.0002)))
            take_profit_atr = float(config_loader.get('TAKE_PROFIT_ATR', self.config.get('take_profit_atr', 2.0)))

            self.config['confidence_threshold'] = confidence_threshold
            self.config['cooldown_period'] = cooldown_period
            self.config['stop_loss_pct'] = stop_loss_pct
            self.config['min_holding_period'] = min_holding_period
            self.config['slippage'] = slippage
            self.config['fee'] = fee
            self.config['take_profit_atr'] = take_profit_atr

            cooldown_counter = 0
            last_trade_was_loss = False
            position = 0
            equity = 100000.0
            equity_curve = []  # Now stores tuples: (equity, action)
            trade_returns = []
            trade_dates = []
            trade_types = []
            trade_pnls = []
            entry_price = 0
            entry_idx = None
            max_equity = equity
            drawdowns = []
            holding_periods = []
            last_trade_idx = None
            long_trades = []
            short_trades = []
            holding_days = 0
            last_trade_month = None
            did_trade_this_month = False
            first_trade_done = False
            last_signal = None
            for i, row in df.iterrows():
                signal = row['signal']
                price = row['close']
                confidence = row['confidence'] if 'confidence' in row else 50
                current_month = i.month if hasattr(i, 'month') else pd.to_datetime(i).month
                action = None
                # --- Enhanced Regime Detection and Volatility/Trend Filters ---
                sma_20 = row['sma_20'] if 'sma_20' in row else None
                sma_50 = row['sma_50'] if 'sma_50' in row else None
                sma_200 = row['sma_200'] if 'sma_200' in row else None
                rsi = row['rsi'] if 'rsi' in row else None
                macd = row['macd'] if 'macd' in row else None
                macd_signal = row['macd_signal'] if 'macd_signal' in row else None
                bb_upper = row['bb_upper'] if 'bb_upper' in row else None
                bb_lower = row['bb_lower'] if 'bb_lower' in row else None
                adx = row['adx'] if 'adx' in row else None
                obv = row['obv'] if 'obv' in row else None
                vwap = row['vwap'] if 'vwap' in row else None
                stoch_k = row['stoch_k'] if 'stoch_k' in row else None
                stoch_d = row['stoch_d'] if 'stoch_d' in row else None
                atr_14 = row['atr_14'] if 'atr_14' in row else None
                # --- Robust Regime Detection ---
                uptrend = False
                downtrend = False
                if price is not None and sma_50 is not None and sma_200 is not None and adx is not None:
                    if price > sma_200 and sma_50 > sma_200 and adx > 20:
                        uptrend = True
                    if price < sma_200 and sma_50 < sma_200 and adx > 25:
                        downtrend = True
                # --- Robust Volatility Filter ---
                volatility_ok = False
                high_volatility = False
                if atr_14 is not None and price is not None:
                    vol_ratio = atr_14 / price
                    if vol_ratio > 0.01:
                        volatility_ok = True
                    if vol_ratio > 0.025:
                        high_volatility = True
                # Multi-indicator confluence (weighted)
                weights = {
                    'trend': 2,
                    'momentum': 2,
                    'volatility': 1,
                    'volume': 1,
                    'mean_reversion': 1,
                    'stoch': 1
                }
                buy_score = 0
                sell_score = 0
                # Trend: price > sma_50 and sma_50 > sma_200 (bullish)
                if price is not None and sma_50 is not None and sma_200 is not None:
                    if price > sma_50 and sma_50 > sma_200:
                        buy_score += weights['trend']
                    elif price < sma_50 and sma_50 < sma_200:
                        sell_score += weights['trend']
                # Momentum: MACD > MACD signal and RSI 40-65 (bullish), MACD < MACD signal and RSI > 70 (bearish)
                if macd is not None and macd_signal is not None and rsi is not None:
                    if macd > macd_signal and 40 < rsi < 65:
                        buy_score += weights['momentum']
                    elif macd < macd_signal and rsi > 70:
                        sell_score += weights['momentum']
                # Volatility: price near lower BB (buy), price near upper BB (sell)
                if price is not None and bb_lower is not None and bb_upper is not None:
                    if price < bb_lower * 1.03:
                        buy_score += weights['volatility']
                    elif price > bb_upper * 0.98:
                        sell_score += weights['volatility']
                # Volume: OBV rising (buy), OBV falling (sell)
                if obv is not None and i != df.index[0]:
                    prev_obv = df.loc[df.index[df.index.get_loc(i)-1], 'obv'] if 'obv' in df.columns and df.index.get_loc(i) > 0 else None
                    if prev_obv is not None:
                        if obv > prev_obv:
                            buy_score += weights['volume']
                        elif obv < prev_obv:
                            sell_score += weights['volume']
                # Mean reversion: price < vwap (buy), price > vwap (sell)
                if price is not None and vwap is not None:
                    if price < vwap * 0.98:
                        buy_score += weights['mean_reversion']
                    elif price > vwap * 1.02:
                        sell_score += weights['mean_reversion']
                # Stochastic: K crosses above D (buy), K crosses below D (sell)
                if stoch_k is not None and stoch_d is not None:
                    if stoch_k > stoch_d and stoch_k < 30:
                        buy_score += weights['stoch']
                    elif stoch_k < stoch_d and stoch_k > 70:
                        sell_score += weights['stoch']
                # --- Enhanced Short Trade Logic ---
                # Only allow short in strong downtrend, high volatility, and high sell_score
                allow_short = False
                if downtrend and high_volatility and sell_score >= 7 and position == 0:
                    allow_short = True
                # Final decision: require regime, volatility, and at least 6 total points for action (was 4)
                if uptrend and volatility_ok and buy_score >= 6 and position == 0:
                    entry_price = price * (1 + slippage + fee)
                    entry_idx = i
                    position = 1
                    holding_days = 0
                    trade_returns.append(0)
                    trade_dates.append(i)
                    trade_types.append('buy')
                    trade_pnls.append(equity)
                    last_trade_idx = i
                    first_trade_done = True
                    did_trade_this_month = True
                    action = 'buy_regime_entry'
                if position == 1 and (not uptrend or not volatility_ok or sell_score >= 6) and holding_days >= min_holding_period:
                    pnl = (price - entry_price) / entry_price - slippage - fee
                    equity *= (1 + pnl)
                    trade_returns.append(pnl)
                    trade_dates.append(i)
                    trade_types.append('sell')
                    trade_pnls.append(equity)
                    long_trades.append(pnl)
                    last_long_profit = pnl
                    if last_trade_idx is not None:
                        holding_periods.append((i - last_trade_idx).days if hasattr(i, 'days') else (i - last_trade_idx))
                    last_trade_idx = i
                    position = 0
                    entry_price = 0
                    holding_days = 0
                    equity_curve.append((equity, 'sell_regime_exit'))
                    max_equity = max(max_equity, equity)
                    drawdowns.append((max_equity - equity) / max_equity)
                    last_signal = signal
                    continue
                # 4. Enhanced Cooldown logic
                # If cooldown is active, log and skip trading
                if self._apply_cooldown(cooldown_counter, position):
                    self.logger.info(f"Cooldown active: {cooldown_counter} days remaining. No new trades allowed at {i}.")
                    cooldown_counter -= 1
                    action = f'cooldown_{cooldown_counter+1}'
                    equity_curve.append((equity, action))
                    last_signal = signal
                    continue
                # Dynamically adjust cooldown based on recent trade outcomes
                # Configurable: number of consecutive losses to trigger cooldown
                consecutive_loss_threshold = int(self.config.get('consecutive_loss_threshold', 3))
                if len(trade_returns) >= consecutive_loss_threshold and all(r < 0 for r in trade_returns[-consecutive_loss_threshold:]) and cooldown_counter == 0:
                    cooldown_counter = min(cooldown_period * 2, 10)  # Cap max cooldown
                    self.logger.warning(f"{consecutive_loss_threshold} consecutive losses. Cooldown increased to {cooldown_counter} at {i}.")
                # Configurable: rapid trade interval (days)
                rapid_trade_interval = int(self.config.get('rapid_trade_interval', 2))
                if len(trade_dates) >= 3 and position == 0:
                    # Performance: use list comprehension for interval calculation
                    try:
                        recent_intervals = [
                            (trade_dates[-idx] - trade_dates[-(idx+1)]).days if hasattr(trade_dates[-idx] - trade_dates[-(idx+1)], 'days') else (trade_dates[-idx] - trade_dates[-(idx+1)])
                            for idx in range(1, 3)
                        ]
                    except Exception as e:
                        self.logger.debug(f"Interval calculation error at {i}: {e}")
                        recent_intervals = []
                    if recent_intervals and all(ri <= rapid_trade_interval for ri in recent_intervals):
                        cooldown_counter = max(cooldown_counter, cooldown_period)
                        self.logger.info(f"Rapid consecutive trades detected at {i}. Cooldown set to {cooldown_counter} (intervals: {recent_intervals}).")
                # Ensure at least one trade per month (buy/sell)
                if last_trade_month is not None and current_month != last_trade_month:
                    if not did_trade_this_month and position == 0 and allow_entry:
                        self.logger.info(f"Monthly entry enforced at {i} (month {current_month}).")
                        entry_price = price * (1 + slippage + fee)
                        entry_idx = i
                        position = 1
                        holding_days = 0
                        trade_returns.append(0)
                        trade_dates.append(i)
                        trade_types.append('buy')
                        trade_pnls.append(equity)
                        last_trade_idx = i
                        first_trade_done = True
                        did_trade_this_month = True
                        action = 'monthly_entry'
                if not first_trade_done and position == 0 and signal == 'sell':
                    signal = 'buy'
                # 2. Multi-indicator confirmation for entry
                allow_entry = self._allow_entry(signal, confidence, position)
                df.at[i, 'allow_entry'] = allow_entry
                if allow_entry:
                    self.logger.info(f"Entry allowed at {i}: signal={signal}, confidence={confidence}")
                # 3. Only allow re-entry after a trade if a new signal is present and confidence is high
                allow_reentry = self._should_reenter(last_signal, signal, confidence, position)
                df.at[i, 'allow_reentry'] = allow_reentry
                if allow_reentry:
                    self.logger.info(f"Re-entry allowed at {i}: last_signal={last_signal}, signal={signal}, confidence={confidence}")
                # 1. Enhanced Dynamic stop-loss (with logging and traceability)
                stop_loss_pct_row = self._dynamic_stop_loss(row, price) if 'atr_14' in row else stop_loss_pct
                df.at[i, 'stop_loss_pct'] = stop_loss_pct_row  # For traceability
                if position != 0 and entry_price is not None and entry_price > 0:
                    holding_days += 1
                    if position == 1:
                        loss = (price - entry_price) / entry_price if price is not None and entry_price is not None else None
                        if loss is not None and stop_loss_pct_row is not None and self._safe_compare(loss, -stop_loss_pct_row, lambda x, y: x < y) and holding_days >= min_holding_period:
                            self.logger.info(f"Stop-loss triggered at {i}: loss={loss:.4f}, threshold={-stop_loss_pct_row:.4f}")
                            pnl = (price - entry_price) / entry_price - slippage - fee
                            equity *= (1 + pnl)
                            trade_returns.append(pnl)
                            trade_dates.append(i)
                            trade_types.append('stop_loss_sell')
                            trade_pnls.append(equity)
                            long_trades.append(pnl)
                            last_long_profit = pnl
                            position = 0
                            entry_price = 0
                            last_trade_idx = i
                            equity_curve.append((equity, 'stop_loss_sell'))
                            max_equity = max(max_equity, equity)
                            drawdowns.append((max_equity - equity) / max_equity)
                            holding_days = 0
                            if pnl < 0:
                                cooldown_counter = cooldown_period
                                last_trade_was_loss = True
                            last_signal = signal
                            continue
                    elif position == -1:
                        loss = (entry_price - price) / entry_price if price is not None and entry_price is not None else None
                        if loss is not None and stop_loss_pct_row is not None and self._safe_compare(loss, -stop_loss_pct_row, lambda x, y: x < y) and holding_days >= min_holding_period:
                            self.logger.info(f"Short stop-loss triggered at {i}: loss={loss:.4f}, threshold={-stop_loss_pct_row:.4f}")
                            pnl = (entry_price - price) / entry_price - slippage - fee
                            equity *= (1 + pnl)
                            trade_returns.append(pnl)
                            trade_dates.append(i)
                            trade_types.append('stop_loss_cover')
                            trade_pnls.append(equity)
                            short_trades.append(pnl)
                            last_short_profit = pnl
                            position = 0
                            entry_price = 0
                            last_trade_idx = i
                            equity_curve.append((equity, 'stop_loss_cover'))
                            max_equity = max(max_equity, equity)
                            drawdowns.append((max_equity - equity) / max_equity)
                            holding_days = 0
                            if pnl < 0:
                                cooldown_counter = cooldown_period
                                last_trade_was_loss = True
                            last_signal = signal
                            continue
                # Dynamic take-profit
                take_profit_atr_row = self._dynamic_take_profit(row, entry_price) if 'atr_14' in row else (take_profit_atr / entry_price if entry_price is not None and entry_price > 0 else 0.20)
                if position == 1 and price is not None and entry_price is not None and take_profit_atr_row is not None and self._safe_compare((price - entry_price) / entry_price, take_profit_atr_row, lambda x, y: x >= y) and holding_days >= min_holding_period:
                    pnl = (price - entry_price) / entry_price - slippage - fee
                    equity *= (1 + pnl)
                    trade_returns.append(pnl)
                    trade_dates.append(i)
                    trade_types.append('take_profit_sell')
                    trade_pnls.append(equity)
                    long_trades.append(pnl)
                    last_long_profit = pnl
                    if last_trade_idx is not None:
                        holding_periods.append((i - last_trade_idx).days if hasattr(i, 'days') else (i - last_trade_idx))
                    last_trade_idx = i
                    position = 0
                    entry_price = 0
                    holding_days = 0
                    equity_curve.append((equity, 'take_profit_sell'))
                    max_equity = max(max_equity, equity)
                    drawdowns.append((max_equity - equity) / max_equity)
                    # Re-entry logic
                    if allow_reentry:
                        entry_price = price * (1 + slippage + fee)
                        entry_idx = i
                        position = 1
                        holding_days = 0
                        trade_returns.append(0)
                        trade_dates.append(i)
                        trade_types.append('buy')
                        trade_pnls.append(equity)
                        last_trade_idx = i
                    last_signal = signal
                    continue
                # Trailing stop: if profit > 10% and price drops 5% from peak, sell
                if position == 1 and holding_days >= min_holding_period:
                    if not hasattr(self, 'peak_price') or position == 0:
                        self.peak_price = entry_price
                    if price is not None and self.peak_price is not None:
                        self.peak_price = max(self.peak_price, price)
                        profit = (price - entry_price) / entry_price if price is not None and entry_price is not None else None
                        drop = (self.peak_price - price) / self.peak_price if self.peak_price is not None and price is not None else None
                        if profit is not None and drop is not None and self._safe_compare(drop, 0.05, lambda x, y: x > y) and self._safe_compare(profit, 0.10, lambda x, y: x > y):
                            pnl = (price - entry_price) / entry_price - slippage - fee
                            equity *= (1 + pnl)
                            trade_returns.append(pnl)
                            trade_dates.append(i)
                            trade_types.append('trailing_stop_sell')
                            trade_pnls.append(equity)
                            long_trades.append(pnl)
                            last_long_profit = pnl
                            position = 0
                            entry_price = 0
                            last_trade_idx = i
                            equity_curve.append((equity, 'trailing_stop_sell'))
                            max_equity = max(max_equity, equity)
                            drawdowns.append((max_equity - equity) / max_equity)
                            holding_days = 0
                            last_signal = signal
                            continue
                # Human-like stop: if loss > 10% after 30 days, close position
                if position == 1 and (price - entry_price) / entry_price < -0.10 and holding_days > 30:
                    pnl = (price - entry_price) / entry_price - slippage - fee
                    equity *= (1 + pnl)
                    trade_returns.append(pnl)
                    trade_dates.append(i)
                    trade_types.append('emotional_stop_loss')
                    trade_pnls.append(equity)
                    long_trades.append(pnl)
                    last_long_profit = pnl
                    if last_trade_idx is not None:
                        holding_periods.append((i - last_trade_idx).days if hasattr(i, 'days') else (i - last_trade_idx))
                    last_trade_idx = i
                    position = 0
                    entry_price = 0
                    holding_days = 0
                    equity_curve.append((equity, 'emotional_stop_loss'))
                    max_equity = max(max_equity, equity)
                    drawdowns.append((max_equity - equity) / max_equity)
                    cooldown_counter = cooldown_period
                    last_trade_was_loss = True
                    continue
                # Only allow short if at least one long trade has pnl > 0.1 (10%)
                if signal == 'sell' and position == 0 and not first_trade_done:
                    continue
                if signal == 'sell' and position == 0:
                    if last_long_profit is None or last_long_profit <= 0.1:
                        continue
                    # --- Only allow short if regime and volatility filters pass ---
                    if not allow_short:
                        continue
                    entry_price = price * (1 - slippage - fee)
                    entry_idx = i
                    position = -1
                    holding_days = 0
                    trade_returns.append(0)
                    trade_dates.append(i)
                    trade_types.append('short')
                    trade_pnls.append(equity)
                    last_trade_idx = i
                    action = 'short_entry'
                # Always close current position on opposite signal, but only if min holding period met
                if position == 1 and signal == 'sell' and holding_days >= min_holding_period:
                    pnl = (price - entry_price) / entry_price - slippage - fee
                    equity *= (1 + pnl)
                    trade_returns.append(pnl)
                    trade_dates.append(i)
                    trade_types.append('sell')
                    trade_pnls.append(equity)
                    long_trades.append(pnl)
                    last_long_profit = pnl
                    if last_trade_idx is not None:
                        holding_periods.append((i - last_trade_idx).days if hasattr(i, 'days') else (i - last_trade_idx))
                    last_trade_idx = i
                    position = 0
                    entry_price = 0
                    holding_days = 0
                    equity_curve.append((equity, 'sell_exit'))
                elif position == -1 and signal == 'buy' and holding_days >= min_holding_period:
                    pnl = (entry_price - price) / entry_price - slippage - fee
                    equity *= (1 + pnl)
                    trade_returns.append(pnl)
                    trade_dates.append(i)
                    trade_types.append('cover')
                    trade_pnls.append(equity)
                    short_trades.append(pnl)
                    if last_trade_idx is not None:
                        holding_periods.append((i - last_trade_idx).days if hasattr(i, 'days') else (i - last_trade_idx))
                    last_trade_idx = i
                    position = 0
                    entry_price = 0
                    holding_days = 0
                    equity_curve.append((equity, 'cover_exit'))
                # If flat, accrue interest on cash
                if position == 0:
                    equity *= (1 + daily_rf_rate)
                    action = 'interest'
                # Always open a new position on every buy/sell signal, even if previous signal was the same
                if signal == 'buy' and position == 0 and confidence > 60:
                    entry_price = price * (1 + slippage + fee)
                    entry_idx = i
                    position = 1
                    holding_days = 0
                    trade_returns.append(0)
                    trade_dates.append(i)
                    trade_types.append('buy')
                    trade_pnls.append(equity)
                    last_trade_idx = i
                    action = 'buy_entry'
                equity_curve.append((equity, action))
                max_equity = max(max_equity, equity)
                drawdowns.append((max_equity - equity) / max_equity)
                last_trade_month = current_month
                if signal in ['buy', 'sell']:
                    did_trade_this_month = True
                # Log every equity-changing event, even if not a trade (for full traceability)
                if len(trade_dates) == 0 or trade_dates[-1] != i:
                    trade_dates.append(i)
                    trade_types.append('hold')
                    trade_returns.append(0)
                    trade_pnls.append(equity)
            # After loop, split equity_curve into two columns for DataFrame
            equity_vals, actions = zip(*equity_curve) if equity_curve else ([],[])
            df['equity_curve'] = list(equity_vals) + [equity]*(len(df)-len(equity_vals))
            df['equity_action'] = list(actions) + ['']*(len(df)-len(actions))
            df['strategy'] = df['equity_curve'].pct_change().fillna(0)
            # Remove 'hold' actions from trade log
            filtered_trade_log = [(d, t, r, p) for d, t, r, p in zip(trade_dates, trade_types, trade_returns, trade_pnls) if t != 'hold']
            if filtered_trade_log:
                trade_dates, trade_types, trade_returns, trade_pnls = zip(*filtered_trade_log)
            else:
                trade_dates, trade_types, trade_returns, trade_pnls = [], [], [], []
            # Defensive checks for empty or single-trade cases
            n_trades = len(trade_returns)
            n_long = len(long_trades)
            n_short = len(short_trades)
            # Sharpe ratio with risk-free rate
            excess_returns = df['strategy'] - daily_rf_rate if 'strategy' in df.columns else np.array([])
            sharpe = (np.mean(excess_returns) / (np.std(excess_returns) + 1e-9) * np.sqrt(252)) if len(excess_returns) > 1 else np.nan
            # Sortino: downside deviation only
            downside = excess_returns[excess_returns < 0] if len(excess_returns) > 0 else np.array([])
            sortino = (np.mean(excess_returns) / (np.std(downside) + 1e-9) * np.sqrt(252)) if len(downside) > 0 else np.nan
            # Metrics (all from start_of_year onward)
            # 1. Total Return: Robust to zero division, NaN, and empty equity curve
            if len(df) > 1 and df['equity_curve'].iloc[0] > 0:
                total_return = df['equity_curve'].iloc[-1] / df['equity_curve'].iloc[0] - 1
            else:
                total_return = 0.0
            # 2. Annual Return: Robust to zero division, NaN, and short periods
            if len(df) > 1:
                if hasattr(df.index[-1] - df.index[0], 'days'):
                    n_years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-6)
                else:
                    n_years = max(len(df)/252, 1e-6)
                try:
                    annual_return = (df['equity_curve'].iloc[-1] / df['equity_curve'].iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else float('nan')
                except Exception:
                    annual_return = float('nan')
            else:
                annual_return = float('nan')
            # 3. Max Drawdown: Robust to empty drawdowns
            max_drawdown = float(np.nanmax(drawdowns)) if drawdowns else 0.0
            # 4. Calmar: handle zero/negative drawdown
            calmar = (total_return / max_drawdown) if max_drawdown > 0 else np.nan
            # 5. Win rates
            win_rate = np.mean([1 if r > 0 else 0 for r in trade_returns]) if trade_returns else 0.0
            gross_win_rate = np.mean([1 if r > 0 else 0 for r in trade_returns]) if n_trades > 0 else np.nan
            net_win_rate = np.mean([1 if (r - slippage - fee) > 0 else 0 for r in trade_returns]) if n_trades > 0 else np.nan
            long_win = np.mean([1 if r > 0 else 0 for r in long_trades]) if n_long > 0 else np.nan
            short_win = np.mean([1 if r > 0 else 0 for r in short_trades]) if n_short > 0 else np.nan
            # 6. Profit factor
            gross_profit = sum([r for r in trade_returns if r > 0])
            gross_loss = -sum([r for r in trade_returns if r < 0])
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
            net_profit = sum([r - slippage - fee for r in trade_returns if r > 0])
            net_loss = -sum([r - slippage - fee for r in trade_returns if r < 0])
            net_profit_factor = (net_profit / net_loss) if net_loss > 0 else np.nan
            # 7. Expectancy
            expectancy = np.mean(trade_returns) if n_trades > 0 else np.nan
            # 8. Median stats
            median_trade = np.median(trade_returns) if n_trades > 0 else np.nan
            median_holding = np.median(holding_periods) if len(holding_periods) > 0 else np.nan
            # 9. Long/Short stats
            long_avg = np.mean(long_trades) if n_long > 0 else np.nan
            short_avg = np.mean(short_trades) if n_short > 0 else np.nan
            long_median = np.median(long_trades) if n_long > 0 else np.nan
            short_median = np.median(short_trades) if n_short > 0 else np.nan
            # Results dictionary (document np.nan for undefined metrics)
            results = {
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe': sharpe,
                'sortino': sortino,
                'calmar': calmar,
                'gross_win_rate': gross_win_rate,
                'net_win_rate': net_win_rate,
                'num_trades': n_trades,
                'avg_trade': np.mean(trade_returns) if n_trades > 0 else np.nan,
                'median_trade': median_trade,
                'avg_holding_period': np.mean(holding_periods) if len(holding_periods) > 0 else np.nan,
                'median_holding_period': median_holding,
                'profit_factor': profit_factor,
                'net_profit_factor': net_profit_factor,
                'expectancy': expectancy,
                'long_win_rate': long_win,
                'short_win_rate': short_win,
                'long_avg_trade': long_avg,
                'short_avg_trade': short_avg,
                'long_median_trade': long_median,
                'short_median_trade': short_median,
                'num_long_trades': n_long,
                'num_short_trades': n_short,
                'note': 'np.nan means metric is undefined due to insufficient data.'
            }
            # Log summary table
            summary_str = pformat(results, indent=2, width=100)
            # Do not save pretty-printed summary as text file
            # if symbol:
            #     with open(f'backtesting_results/{symbol}_backtest_summary.txt', 'w') as f:
            #         f.write(summary_str)
            # Save results
            if symbol:
                os.makedirs('backtesting_results', exist_ok=True)
                df[['equity_curve', 'strategy', 'signal']].to_csv(f'backtesting_results/{symbol}_backtest.csv')
                # Enhanced trade log details
                trade_log_rows = []
                for idx in range(len(trade_dates)):
                    t_type = trade_types[idx]
                    t_date = trade_dates[idx]
                    t_pnl = trade_returns[idx]
                    t_equity = trade_pnls[idx]
                    # Determine open/close for each trade
                    if t_type in ['buy', 'short', 'buy_entry', 'monthly_entry', 'buy_regime_entry']:
                        open_date = t_date
                        open_price = df.loc[open_date, 'close'] if open_date in df.index and 'close' in df.columns else None
                        # Find corresponding close
                        close_idx = None
                        close_type = None
                        for j in range(idx+1, len(trade_types)):
                            if trade_types[j] in ['sell', 'cover', 'take_profit_sell', 'trailing_stop_sell', 'stop_loss_sell', 'emotional_stop_loss', 'sell_regime_exit', 'sell_exit', 'stop_loss_cover', 'cover_exit']:
                                close_idx = j
                                close_type = trade_types[j]
                                break
                        # Get reason for open and close if available
                        open_reason = df.loc[open_date, 'reason'] if 'reason' in df.columns and open_date in df.index else ''
                        close_reason = df.loc[trade_dates[close_idx], 'reason'] if close_idx is not None and 'reason' in df.columns and trade_dates[close_idx] in df.index else ''
                        if close_idx is not None:
                            close_date = trade_dates[close_idx]
                            close_price = df.loc[close_date, 'close'] if close_date in df.index and 'close' in df.columns else None
                            holding_period = (close_date - open_date).days if hasattr(close_date - open_date, 'days') else (close_date - open_date)
                            # Explicit return calculation for closed trades
                            trade_return = None
                            if open_price is not None and close_price is not None and open_price != 0:
                                if t_type in ['buy', 'buy_entry', 'monthly_entry', 'buy_regime_entry']:
                                    trade_return = (close_price - open_price) / open_price
                                elif t_type in ['short']:
                                    trade_return = (open_price - close_price) / open_price
                                else:
                                    trade_return = trade_returns[close_idx]
                            else:
                                trade_return = trade_returns[close_idx]
                            trade_log_rows.append({
                                'date': open_date,
                                'open_date': open_date,
                                'close_date': close_date,
                                'open_price': open_price,
                                'close_price': close_price,
                                'holding_period': holding_period,
                                'type': t_type,
                                'close_type': close_type,
                                'pnl': trade_returns[close_idx],
                                'return': trade_return,
                                'equity': trade_pnls[close_idx],
                                'open_reason': open_reason,
                                'close_reason': close_reason
                            })
                        else:
                            # Handle open trades at the end of the backtest (unrealized P&L)
                            close_date = df.index[-1]
                            close_price = df.loc[close_date, 'close'] if close_date in df.index and 'close' in df.columns else None
                            holding_period = (close_date - open_date).days if hasattr(close_date - open_date, 'days') else (close_date - open_date)
                            trade_return = None
                            if open_price is not None and close_price is not None and open_price != 0:
                                if t_type in ['buy', 'buy_entry', 'monthly_entry', 'buy_regime_entry']:
                                    trade_return = (close_price - open_price) / open_price
                                elif t_type in ['short']:
                                    trade_return = (open_price - close_price) / open_price
                                else:
                                    trade_return = 0
                            else:
                                trade_return = 0
                            trade_log_rows.append({
                                'date': open_date,
                                'open_date': open_date,
                                'close_date': close_date,
                                'open_price': open_price,
                                'close_price': close_price,
                                'holding_period': holding_period,
                                'type': t_type,
                                'close_type': 'open_at_end',
                                'pnl': trade_return,  # Use unrealized P&L for open trades
                                'return': trade_return,
                                'equity': t_equity,
                                'open_reason': open_reason,
                                'close_reason': ''
                            })
                # Save enhanced trade log
                trade_log_df = pd.DataFrame(trade_log_rows)
                # --- Trade log optimization for readability ---
                # 1. Order columns logically
                column_order = [
                    'open_date', 'close_date', 'holding_period',
                    'open_price', 'close_price',
                    'entry_type', 'exit_type',
                    'pnl', 'return', 'equity',
                    'open_reason', 'close_reason'
                ]
                # 2. Rename columns for clarity
                trade_log_df = trade_log_df.rename(columns={
                    'type': 'entry_type',
                    'close_type': 'exit_type',
                    'pnl': 'pnl',
                    'return': 'return',
                    'equity': 'equity',
                    'open_reason': 'open_reason',
                    'close_reason': 'close_reason',
                })
                # 3. Drop redundant columns
                for col in ['date']:
                    if col in trade_log_df.columns:
                        trade_log_df = trade_log_df.drop(columns=[col])
                # 4. Format/round float values
                float_cols = ['open_price', 'close_price', 'pnl', 'return', 'equity']
                for col in float_cols:
                    if col in trade_log_df.columns:
                        trade_log_df[col] = trade_log_df[col].apply(lambda x: round(x, 4) if pd.notnull(x) else x)
                if 'holding_period' in trade_log_df.columns:
                    trade_log_df['holding_period'] = trade_log_df['holding_period'].apply(lambda x: int(x) if pd.notnull(x) else x)
                # 5. Ensure reasons are always objects (dicts)
                def reason_to_object(reason):
                    if not reason:
                        return {}
                    if isinstance(reason, dict):
                        return reason
                    if isinstance(reason, str):
                        s = reason.strip()
                        if s.startswith('{') and s.endswith('}'):  # Try parse JSON
                            try:
                                obj = json.loads(s)
                                if isinstance(obj, dict):
                                    return obj
                            except Exception:
                                pass
                        # Fallback: wrap as dict
                        return {"reason": s}
                    # Fallback: wrap as dict
                    return {"reason": str(reason)}
                for col in ['open_reason', 'close_reason']:
                    if col in trade_log_df.columns:
                        trade_log_df[col] = trade_log_df[col].apply(reason_to_object)
                # 6. Reorder columns
                trade_log_df = trade_log_df[[c for c in column_order if c in trade_log_df.columns]]
                # 7. Save as pretty-printed JSON
                try:
                    # Convert all non-JSON-serializable types to string or float
                    def make_json_serializable(val):
                        import numpy as np
                        import pandas as pd
                        if isinstance(val, (pd.Timestamp, pd.Timedelta)):
                            return str(val)
                        if isinstance(val, (np.generic,)):
                            return float(val)
                        return val
                    # Use DataFrame.astype(object) and .map for elementwise mapping
                    trade_log_json = trade_log_df.astype(object).map(make_json_serializable).to_dict(orient='records')
                    with open(f'backtesting_results/{symbol}_trade_log.json', 'w', encoding='utf-8') as f:
                        json.dump(trade_log_json, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Error saving trade log JSON: {e}")
                try:
                    with open(f'backtesting_results/{symbol}_backtest_summary.json', 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                except Exception as e:
                    print(f"Error saving summary JSON: {e}")

            # Save enhanced trade log and summary
            save_trade_log_and_summary(trade_log_df, results, symbol)
            self.logger.info(f"Backtest complete for {symbol}")
            return results
        except Exception as e:
            self.logger.error(f"Error in backtesting: {e}")
            return {}
