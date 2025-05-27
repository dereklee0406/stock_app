"""
Signal generation module for Stock Analysis Tool.
Generates trading signals based on technical indicators and patterns.
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import logging
from typing import Optional
from logs.logger_setup import LoggerSetup
from config.config_loader import ConfigLoader

class SignalGenerator:
    """
    Generates buy/sell/hold signals from indicator data and plots price, EMA, and Bollinger Bands with buy/sell prices.
    Now populates the 'reason' column with a structured JSON summary of which indicators contributed most to each signal.
    Uses vectorized DataFrame.apply for performance, and indicator logic is extensible via a mapping.
    Always provides a reason for 'hold'.
    """
    def __init__(self, config: dict = None, log_level: str = 'INFO'):
        self.config = config or {}
        self.logger = LoggerSetup(log_level=log_level).get_logger()

    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        if df.empty:
            self.logger.error("No data for signal generation.")
            return df
        result_df = df.copy()
        config_loader = ConfigLoader()

        def safe_float(val, default):
            try:
                if val is None or val == '':
                    return default
                return float(val)
            except Exception:
                return default

        def safe_int(val, default):
            try:
                if val is None or val == '':
                    return default
                return int(float(val))
            except Exception:
                return default

        # --- Indicator weights and thresholds from config/.env ---
        weights = {
            'ema_20': safe_float(config_loader.get('WEIGHT_EMA20', 1.2), 1.2),
            'sma_20': safe_float(config_loader.get('WEIGHT_SMA20', 1.0), 1.0),
            'sma_50': safe_float(config_loader.get('WEIGHT_SMA50', 1.0), 1.0),
            'macd': safe_float(config_loader.get('WEIGHT_MACD', 1.2), 1.2),
            'rsi': safe_float(config_loader.get('WEIGHT_RSI', 1.0), 1.0),
            'bb': safe_float(config_loader.get('WEIGHT_BB', 1.0), 1.0),
            'stoch': safe_float(config_loader.get('WEIGHT_STOCH', 0.8), 0.8),
            'adx': safe_float(config_loader.get('WEIGHT_ADX', 0.8), 0.8),
            'obv': safe_float(config_loader.get('WEIGHT_OBV', 0.6), 0.6),
            'vwap': safe_float(config_loader.get('WEIGHT_VWAP', 1.0), 1.0),
            'ichimoku': safe_float(config_loader.get('WEIGHT_ICHIMOKU', 1.0), 1.0),
            'volatility': safe_float(config_loader.get('WEIGHT_VOLATILITY', 0.7), 0.7),
            'fibo': safe_float(config_loader.get('WEIGHT_FIBO', 0.5), 0.5),
        }
        volume_conf = safe_int(config_loader.get('VOLUME_CONF', 10), 10)
        mtf_conf = safe_int(config_loader.get('MTF_CONF', 10), 10)
        confluence_mult = safe_int(config_loader.get('CONFLUENCE_MULT', 10), 10)
        hist_acc_mult = safe_int(config_loader.get('HIST_ACC_MULT', 10), 10)
        vol_adj = safe_int(config_loader.get('VOL_ADJ', -10), -10)
        min_score = safe_int(config_loader.get('MIN_SCORE', 4), 4)
        min_holding_period = self.config.get('min_holding_period', 5)

        # --- Indicator logic mapping for extensibility ---
        def indicator_logic(row, prev_state):
            close = row.get('close', 0)
            indicators = {
                'sma_20': row.get('sma_20'),
                'sma_50': row.get('sma_50'),
                'ema_20': row.get('ema_20'),
                'ema_50': row.get('ema_50'),
                'rsi': row.get('rsi'),
                'macd': row.get('macd'),
                'macd_signal': row.get('macd_signal'),
                'bb_upper': row.get('bb_upper'),
                'bb_lower': row.get('bb_lower'),
                'stoch_k': row.get('stoch_k'),
                'stoch_d': row.get('stoch_d'),
                'adx': row.get('adx'),
                'obv': row.get('obv'),
                'volume': row.get('volume'),
                'volatility': row.get('rolling_std'),
                'vwap': row.get('vwap'),
                'ichimoku_a': row.get('ichimoku_a'),
                'ichimoku_b': row.get('ichimoku_b'),
                'fibo_0.382': row.get('fibo_0.382'),
                'fibo_0.618': row.get('fibo_0.618'),
            }
            buy_votes, buy_weight, sell_votes, sell_weight = 0, 0, 0, 0
            buy_reasons, sell_reasons, hold_reasons = [], [], []
            # --- Pattern recognition functions ---
            def is_bullish_engulfing(idx):
                if idx == 0 or 'open' not in df.columns or 'close' not in df.columns:
                    return False
                prev = df.iloc[idx-1]
                curr = df.iloc[idx]
                return (
                    prev['close'] < prev['open'] and
                    curr['close'] > curr['open'] and
                    curr['close'] > prev['open'] and
                    curr['open'] < prev['close']
                )
            def is_bearish_engulfing(idx):
                if idx == 0 or 'open' not in df.columns or 'close' not in df.columns:
                    return False
                prev = df.iloc[idx-1]
                curr = df.iloc[idx]
                return (
                    prev['close'] > prev['open'] and
                    curr['close'] < curr['open'] and
                    curr['open'] > prev['close'] and
                    curr['close'] < prev['open']
                )
            def is_hammer(idx):
                if 'open' not in df.columns or 'close' not in df.columns or 'low' not in df.columns or 'high' not in df.columns:
                    return False
                curr = df.iloc[idx]
                body = abs(curr['close'] - curr['open'])
                lower_shadow = curr['open'] - curr['low'] if curr['open'] > curr['close'] else curr['close'] - curr['low']
                upper_shadow = curr['high'] - max(curr['close'], curr['open'])
                return lower_shadow > 2 * body and upper_shadow < body
            def is_doji(idx):
                if 'open' not in df.columns or 'close' not in df.columns:
                    return False
                curr = df.iloc[idx]
                return abs(curr['close'] - curr['open']) < 0.1 * (curr['high'] - curr['low'])
            # --- Indicator logic mapping ---
            logic = [
                # EMA
                (lambda: indicators['ema_20'] is not None and close > indicators['ema_20'], 'buy', 'Close > EMA20', weights['ema_20']),
                (lambda: indicators['ema_20'] is not None and close < indicators['ema_20'], 'sell', 'Close < EMA20', weights['ema_20']),
                # SMA
                (lambda: indicators['sma_20'] is not None and indicators['sma_50'] is not None and indicators['sma_20'] > indicators['sma_50'], 'buy', 'SMA20 > SMA50', weights['sma_20']),
                (lambda: indicators['sma_20'] is not None and indicators['sma_50'] is not None and indicators['sma_20'] < indicators['sma_50'], 'sell', 'SMA20 < SMA50', weights['sma_20']),
                # MACD
                (lambda: indicators['macd'] is not None and indicators['macd_signal'] is not None and indicators['macd'] > indicators['macd_signal'], 'buy', 'MACD > Signal', weights['macd']),
                (lambda: indicators['macd'] is not None and indicators['macd_signal'] is not None and indicators['macd'] < indicators['macd_signal'], 'sell', 'MACD < Signal', weights['macd']),
                # RSI
                (lambda: indicators['rsi'] is not None and indicators['rsi'] < 45, 'buy', 'RSI < 45', weights['rsi']),
                (lambda: indicators['rsi'] is not None and indicators['rsi'] > 55, 'sell', 'RSI > 55', weights['rsi']),
                # BB
                (lambda: indicators['bb_lower'] is not None and close < indicators['bb_lower'], 'buy', 'Close < BB Lower', weights['bb']),
                (lambda: indicators['bb_upper'] is not None and close > indicators['bb_upper'], 'sell', 'Close > BB Upper', weights['bb']),
                # Stoch
                (lambda: indicators['stoch_k'] is not None and indicators['stoch_d'] is not None and indicators['stoch_k'] < 20 and indicators['stoch_d'] < 20, 'buy', 'Stoch < 20', weights['stoch']),
                (lambda: indicators['stoch_k'] is not None and indicators['stoch_d'] is not None and indicators['stoch_k'] > 80 and indicators['stoch_d'] > 80, 'sell', 'Stoch > 80', weights['stoch']),
                # ADX
                (lambda: indicators['adx'] is not None and indicators['adx'] > 20 and prev_state['buy_weight'] > prev_state['sell_weight'], 'buy', 'ADX > 20 (trend strength, buy)', weights['adx']),
                (lambda: indicators['adx'] is not None and indicators['adx'] > 20 and prev_state['sell_weight'] > prev_state['buy_weight'], 'sell', 'ADX > 20 (trend strength, sell)', weights['adx']),
                # OBV
                (lambda: indicators['obv'] is not None and indicators['obv'] > 0, 'buy', 'OBV > 0', weights['obv']),
                (lambda: indicators['obv'] is not None and indicators['obv'] < 0, 'sell', 'OBV < 0', weights['obv']),
                # VWAP
                (lambda: indicators['vwap'] is not None and close > indicators['vwap'], 'buy', 'Close > VWAP', weights['vwap']),
                (lambda: indicators['vwap'] is not None and close < indicators['vwap'], 'sell', 'Close < VWAP', weights['vwap']),
                # Ichimoku
                (lambda: indicators['ichimoku_a'] is not None and indicators['ichimoku_b'] is not None and close > indicators['ichimoku_a'] and close > indicators['ichimoku_b'], 'buy', 'Close > Ichimoku A/B', weights['ichimoku']),
                (lambda: indicators['ichimoku_a'] is not None and indicators['ichimoku_b'] is not None and close < indicators['ichimoku_a'] and close < indicators['ichimoku_b'], 'sell', 'Close < Ichimoku A/B', weights['ichimoku']),
                # Volatility
                (lambda: indicators['volatility'] is not None and indicators['volatility'] < df['close'].rolling(20).std().quantile(0.3), 'buy', 'Low Volatility', weights['volatility']),
                (lambda: indicators['volatility'] is not None and indicators['volatility'] > df['close'].rolling(20).std().quantile(0.7), 'sell', 'High Volatility', weights['volatility']),
                # Fibo
                (lambda: indicators['fibo_0.382'] is not None and abs(close - indicators['fibo_0.382'])/close < 0.01, 'buy', 'Near Fibo 0.382', weights['fibo']),
                (lambda: indicators['fibo_0.618'] is not None and abs(close - indicators['fibo_0.618'])/close < 0.01, 'sell', 'Near Fibo 0.618', weights['fibo']),
            ]
            for cond, typ, reason, w in logic:
                if cond():
                    if typ == 'buy':
                        buy_votes += 1
                        buy_weight += w
                        buy_reasons.append(reason)
                    elif typ == 'sell':
                        sell_votes += 1
                        sell_weight += w
                        sell_reasons.append(reason)
            # --- Pattern recognition logic ---
            idx = row.name if hasattr(row, 'name') else None
            if idx is not None and isinstance(idx, (int, np.integer)):
                if is_bullish_engulfing(idx):
                    buy_reasons.append('Bullish Engulfing Pattern')
                    buy_weight += 0.7
                if is_bearish_engulfing(idx):
                    sell_reasons.append('Bearish Engulfing Pattern')
                    sell_weight += 0.7
                if is_hammer(idx):
                    buy_reasons.append('Hammer Pattern')
                    buy_weight += 0.5
                if is_doji(idx):
                    hold_reasons.append('Doji Pattern (Indecision)')
            # Volume profile
            if indicators['volume'] is not None and 'volume' in df.columns:
                vol_q = df['volume'].quantile(0.7)
                if indicators['volume'] > vol_q:
                    buy_reasons.append('High Volume')
            # MTF (placeholder for true MTF)
            mtf_used = False
            if indicators['ema_20'] is not None and indicators['ema_50'] is not None:
                if (close > indicators['ema_20']) and (indicators['ema_20'] > indicators['ema_50']):
                    buy_reasons.append('MTF Bullish (proxy)')
                    mtf_used = True
            # Compose reason as JSON
            reason_dict = {
                'buy_reasons': buy_reasons,
                'sell_reasons': sell_reasons,
                'hold_reasons': [],
                'top_buy': buy_reasons[:3],
                'top_sell': sell_reasons[:3],
                'mtf_used': mtf_used
            }
            # Hold reason logic
            if not buy_reasons and not sell_reasons:
                hold_reasons.append('No strong consensus')
            if abs(buy_weight - sell_weight) < 0.1:
                hold_reasons.append('Buy/Sell weights nearly equal')
            reason_dict['hold_reasons'] = hold_reasons
            return buy_votes, buy_weight, sell_votes, sell_weight, reason_dict

        # Vectorized apply
        prev_state = {'buy_weight': 0, 'sell_weight': 0}

        def row_apply(row):
            buy_votes, buy_weight, sell_votes, sell_weight, reason_dict = indicator_logic(row, prev_state)
            confidence = int(
                40 + 10 * max(buy_weight, sell_weight) + volume_conf + mtf_conf + confluence_mult * max(buy_votes, sell_votes) + hist_acc_mult + vol_adj
            )
            confidence = max(0, min(confidence, 100))
            signal = 'hold'
            allow_new_signal = prev_state.get('holding_counter', 0) >= min_holding_period or prev_state.get('last_position', 'none') == 'none'
            if buy_votes >= min_score and prev_state.get('last_position', 'none') != 'long' and allow_new_signal and confidence > 40:
                signal = 'buy'
                prev_state['last_position'] = 'long'
                prev_state['holding_counter'] = 0
            elif sell_votes >= min_score and prev_state.get('last_position', 'none') != 'short' and allow_new_signal and confidence > 40:
                signal = 'sell'
                prev_state['last_position'] = 'short'
                prev_state['holding_counter'] = 0
            else:
                signal = 'hold'
                prev_state['holding_counter'] = prev_state.get('holding_counter', 0) + 1
            # Store reason as dict, not JSON string
            return pd.Series({'signal': signal, 'confidence': confidence, 'reason': reason_dict})

        result = df.apply(row_apply, axis=1)
        result_df['signal'] = result['signal']
        result_df['confidence'] = result['confidence']
        result_df['reason'] = result['reason']

        if symbol:
            os.makedirs('stock_signal', exist_ok=True)
            # When saving to CSV, serialize reason as JSON string
            result_df_to_save = result_df.copy()
            result_df_to_save['reason'] = result_df_to_save['reason'].apply(json.dumps)
            result_df_to_save.to_csv(f'stock_signal/{symbol}_signal_history.csv')
            plot_signals(result_df, symbol, log_level=self.logger.level)
        self.logger.info(f"Signals generated and saved for {symbol}")
        return result_df

def _plot_signal_markers(
    ax: plt.Axes,
    signals: pd.DataFrame,
    df: pd.DataFrame,
    signal_type: str,
    marker: str,
    color: str,
    y_offset: int
) -> None:
    """Helper to plot buy/sell markers and confidence annotations."""
    mask = signals['signal'] == signal_type
    idxs = signals.index[mask]
    if not idxs.empty:
        ax.scatter(
            idxs,
            df.loc[idxs, 'close'],
            marker=marker,
            color=color,
            label=signal_type.capitalize(),
            s=120,
            zorder=5,
            edgecolor='black',
            linewidths=1.5
        )
        for idx in idxs:
            conf = int(signals.at[idx, 'confidence']) if 'confidence' in signals.columns else 0
            ax.annotate(
                f"{conf}",
                (mdates.date2num(idx), df.loc[idx, 'close']),
                textcoords="offset points",
                xytext=(24, y_offset),
                ha='left',
                fontsize=9,
                color=color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, lw=1, alpha=0.85)
            )

def _plot_overlay_indicators(ax: plt.Axes, df: pd.DataFrame, overlays: dict) -> None:
    """Plot optional overlays: EMA20, EMA50, Bollinger Bands."""
    if overlays.get('ema20', True) and 'ema_20' in df.columns:
        ax.plot(df.index, df['ema_20'], label='EMA 20', color='#00bcd4', linestyle=':', linewidth=1.5)
    if overlays.get('ema50', True) and 'ema_50' in df.columns:
        ax.plot(df.index, df['ema_50'], label='EMA 50', color='#e91e63', linestyle=':', linewidth=1.5)
    if overlays.get('bb', True) and 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        ax.plot(df.index, df['bb_upper'], color='#1976d2', linestyle='-', linewidth=1.2, label='BB Upper')
        ax.plot(df.index, df['bb_lower'], color='#e74c3c', linestyle='-', linewidth=1.2, label='BB Lower')
        ax.fill_between(df.index, df['bb_upper'], df['bb_lower'], color='#90caf9', alpha=0.18, label='Bollinger Bands')

def _plot_signal_background(ax: plt.Axes, signals: pd.DataFrame) -> None:
    """Shade background for buy/sell/hold regions."""
    for idx, row in signals.iterrows():
        color = None
        if row['signal'] == 'buy':
            color = '#d0f5e8'
        elif row['signal'] == 'sell':
            color = '#fbeee6'
        elif row['signal'] == 'hold':
            color = '#f4f4f4'
        if color:
            ax.axvspan(idx, idx, color=color, alpha=0.18, linewidth=0)

def plot_signals(
    df: pd.DataFrame,
    symbol: str,
    start_date: Optional[str] = None,
    log_level: str = 'INFO',
    show_reasons: bool = False,
    overlays: Optional[dict] = None,
    show_background: bool = True
) -> None:
    """
    Plots price, SMA, and buy/sell signals with confidence annotations only (no reasons).
    Optionally overlays EMA20, EMA50, Bollinger Bands, and background shading for buy/sell/hold.
    Saves the chart to stock_signal/{symbol}_ta_comprehensive_signals_chart.png.
    Limits x-axis ticks if there are too many signals to avoid clutter.
    Only shows legend if there are labeled artists.
    overlays: dict with keys 'ema20', 'ema50', 'bb' (all bool, default True)
    show_background: bool, whether to shade buy/sell/hold regions
    """
    # Ensure log_level is always a string name
    if isinstance(log_level, int):
        log_level = logging.getLevelName(log_level)
    logger = LoggerSetup(log_level=log_level).get_logger()
    try:
        # Defensive style application
        import matplotlib
        available_styles = matplotlib.style.available
        style = 'seaborn-darkgrid' if 'seaborn-darkgrid' in available_styles else 'default'
        try:
            plt.style.use(style)
        except Exception as e:
            logger.warning(f"Style '{style}' failed: {e}. Falling back to 'default'.")
            plt.style.use('default')

        # Date filtering
        if start_date is None:
            start_date = os.getenv('START_DATE', '2025-01-01')
        start_of_period = pd.to_datetime(start_date)
        if not df.empty and hasattr(df.index, 'max'):
            df = df[df.index >= start_of_period]
        signals_path = f'stock_signal/{symbol}_signal_history.csv'
        if not os.path.exists(signals_path):
            logger.error(f'Signal history not found: {signals_path}')
            return
        signals = pd.read_csv(signals_path, index_col=0, parse_dates=True)
        if not signals.empty and hasattr(signals.index, 'max'):
            signals = signals[signals.index >= start_of_period]

        plt.figure(figsize=(18, 8))
        ax = plt.gca()
        plt.plot(df.index, df['close'], label='Close', color='#222', linewidth=2)
        if 'sma_20' in df.columns:
            plt.plot(df.index, df['sma_20'], label='SMA 20', color='#1976d2', linestyle='--', linewidth=1.5)
        # Overlays
        if overlays is None:
            overlays = {'ema20': True, 'ema50': True, 'bb': True}
        _plot_overlay_indicators(ax, df, overlays)
        # Background shading
        if show_background:
            _plot_signal_background(ax, signals)
        # Use helper for buy/sell marker plotting
        _plot_signal_markers(ax, signals, df, 'buy', '^', '#27ae60', -10)
        _plot_signal_markers(ax, signals, df, 'sell', 'v', '#e74c3c', 10)

        # Signal stats annotation
        buy = signals[signals['signal'] == 'buy']
        sell = signals[signals['signal'] == 'sell']
        buy_count = len(buy)
        sell_count = len(sell)
        buy_conf = buy['confidence'].mean() if not buy.empty else 0
        sell_conf = sell['confidence'].mean() if not sell.empty else 0
        stats_text = (
            f"Buy signals: {buy_count} (avg conf: {buy_conf:.1f})\n"
            f"Sell signals: {sell_count} (avg conf: {sell_conf:.1f})"
        )
        plt.gca().text(
            0.99, 0.98, stats_text,
            transform=plt.gca().transAxes,
            fontsize=16, fontweight='bold', color='#333',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7fa', edgecolor='#888', alpha=0.92)
        )
        plt.title(f'{symbol} Price, SMA 20, Buy/Sell Signals', fontsize=18, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price', fontsize=14)
        # Only show legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=13, loc='best', frameon=True)
        plt.grid(True, alpha=0.18)
        plt.tight_layout()
        # X-axis formatting: only show ticks for buy/sell signal dates, but limit if too many
        signal_dates = list(set(signals[signals['signal'] == 'buy'].index).union(set(signals[signals['signal'] == 'sell'].index)))
        signal_dates = sorted(signal_dates)
        max_ticks = 30
        if signal_dates:
            if len(signal_dates) > max_ticks:
                # Downsample signal dates for ticks
                step = max(1, len(signal_dates) // max_ticks)
                tick_dates = signal_dates[::step]
            else:
                tick_dates = signal_dates
            ax.set_xticks(tick_dates)
            ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in tick_dates], rotation=30, ha='right', fontsize=12)
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=12)
        os.makedirs('stock_signal', exist_ok=True)
        out_path = f'stock_signal/{symbol}_ta_comprehensive_signals_chart.png'
        plt.savefig(out_path, dpi=180)
        plt.close()
        logger.info(f'Detailed signal chart saved: {out_path}')
    except Exception as e:
        logger.error(f'Error generating detailed signal chart: {e}')
