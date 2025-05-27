"""
Technical analysis module for Stock Analysis Tool.
Calculates technical indicators on stock data.
"""
import pandas as pd
import ta
import os
import matplotlib.pyplot as plt
from logs.logger_setup import LoggerSetup

class TechnicalAnalysis:
    """
    Calculates and appends technical indicators to stock data.
    """
    def __init__(self, config: dict = None, log_level: str = 'INFO'):
        self.config = config or {}
        self.logger = LoggerSetup(log_level=log_level).get_logger()

    def calculate_indicators(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Calculates indicators (SMA, EMA, RSI, MACD, etc.) and appends to DataFrame.
        Saves the result as {stock}_indicators.csv in stock_indicators/.
        """
        if df.empty or df[['close', 'volume']].isnull().all().any():
            self.logger.error("Insufficient data for indicator calculation.")
            return df
        try:
            # Cache config lookups
            ma_windows = self.config.get('ma_windows', [20, 50, 200])
            rsi_period = self.config.get('rsi_period', 14)
            bb_window = self.config.get('bb_window', 20)
            stoch_window = self.config.get('stoch_window', 14)
            adx_window = self.config.get('adx_window', 14)
            std_window = self.config.get('std_window', 20)
            atr_window = self.config.get('atr_window', 14)
            mfi_window = self.config.get('mfi_window', 14)
            interval = self.config.get('interval', '')

            # Moving Averages
            for window in ma_windows:
                df[f'sma_{window}'] = ta.trend.sma_indicator(df['close'], window=window)
                df[f'ema_{window}'] = ta.trend.ema_indicator(df['close'], window=window)
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=rsi_period)
            # MACD
            df['macd'] = ta.trend.macd(df['close'])
            df['macd_signal'] = ta.trend.macd_signal(df['close'])
            df['macd_diff'] = ta.trend.macd_diff(df['close'])
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=bb_window)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=stoch_window)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            df['stoch_j'] = 3 * df['stoch_k'] - 2 * df['stoch_d']
            # ADX
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=adx_window)
            # OBV
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            # VWAP (intraday only)
            if '1min' in interval:
                df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
            # Ichimoku
            ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            # Rolling std
            df['rolling_std'] = df['close'].rolling(window=std_window).std()
            # Fibonacci levels (0, 0.382, 0.5, 0.618, 1)
            min_price = df['close'].min()
            max_price = df['close'].max()
            fibo_levels = [0, 0.382, 0.5, 0.618, 1]
            for level in fibo_levels:
                df[f'fibo_{level}'] = min_price + (max_price - min_price) * level
            # ATR (Average True Range)
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=atr_window)
            # MFI (Money Flow Index)
            df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=mfi_window)
            # Save indicators
            if symbol:
                os.makedirs('stock_indicators', exist_ok=True)
                df.to_csv(f'stock_indicators/{symbol}_indicators.csv')
            self.logger.info(f"Indicators calculated and saved for {symbol}")
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")
        return df

class ChartGenerator:
    """
    Generates and saves charts for price and indicators.
    """
    def __init__(self, log_level: str = 'INFO'):
        self.logger = LoggerSetup(log_level=log_level).get_logger()

    def _style_axis(self, ax, ylabel, color='#222', grid_alpha=0.10, grid_color='#bbb'):
        ax.set_ylabel(ylabel, fontsize=13, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.grid(True, alpha=grid_alpha, color=grid_color)

    def _style_legend(self, ax, loc='upper left'):
        leg = ax.legend(loc=loc, fontsize=11, frameon=True)
        if leg:
            leg.get_frame().set_facecolor('#fff')
            leg.get_frame().set_edgecolor('#888')
            for text in leg.get_texts():
                text.set_color('#222')

    def _draw_summary_box(self, fig, stoch_k_latest, stoch_d_latest, stoch_j_latest, mfi_latest):
        summary_lines = []
        if stoch_k_latest is not None:
            summary_lines.append(f"%K: {stoch_k_latest:.2f}")
        if stoch_d_latest is not None:
            summary_lines.append(f"%D: {stoch_d_latest:.2f}")
        if stoch_j_latest is not None:
            summary_lines.append(f"%J: {stoch_j_latest:.2f}")
        if mfi_latest is not None:
            summary_lines.append(f"MFI: {mfi_latest:.2f}")
        if summary_lines:
            fig.text(0.995, 0.18, '\n'.join(summary_lines), fontsize=12, color='#222', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.4', fc='#fff', ec='#888', lw=1, alpha=0.85))

    def _plot_price_ma_bb_ema_sr(self, ax_price, df, symbol):
        # Price, MA, BB, EMA, S/R plotting
        ax_price.plot(df.index, df['close'], label='Close', color='#222', linewidth=2.2)
        if 'sma_20' in df.columns:
            ax_price.plot(df.index, df['sma_20'], label='SMA 20', color='#1976d2', linestyle='--', linewidth=1.8)
        if 'sma_50' in df.columns:
            ax_price.plot(df.index, df['sma_50'], label='SMA 50', color='#ffb300', linestyle='--', linewidth=1.8)
        # Plot all EMA lines and collect latest values for annotation
        ema_cols = [col for col in df.columns if col.startswith('ema_')]
        ema_colors = ['#009688', '#e91e63', '#00bcd4', '#ff9800', '#8bc34a', '#9c27b0', '#607d8b']
        ema_labels = []
        for i, ema_col in enumerate(sorted(ema_cols, key=lambda x: int(x.split('_')[1]))):
            ax_price.plot(
                df.index, df[ema_col],
                label=ema_col.upper(), linestyle=':', linewidth=1.5, color=ema_colors[i % len(ema_colors)]
            )
            latest_ema = df[ema_col].iloc[-1]
            ema_labels.append(f"{ema_col.upper()}: {latest_ema:.2f}")
        if ema_labels:
            ax_price.text(
                0.995, 0.01, '\n'.join(ema_labels),
                transform=ax_price.transAxes, fontsize=11, color='#e0e0e0', fontweight='bold',
                va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', fc='black', ec='#888', lw=1, alpha=0.7)
            )
        # Bollinger Bands
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            ax_price.plot(df.index, df['bb_upper'], color='#1976d2', linestyle='-', linewidth=2.2, label='BB Upper')
            ax_price.plot(df.index, df['bb_lower'], color='#e74c3c', linestyle='-', linewidth=2.2, label='BB Lower')
            ax_price.fill_between(df.index, df['bb_upper'], df['bb_lower'], color='#90caf9', alpha=0.28, label='Bollinger Bands')
        # Support/Resistance
        from scipy.signal import argrelextrema
        import numpy as np
        window = 10
        close_np = df['close'].values
        support_idx = argrelextrema(close_np, np.less_equal, order=window)[0]
        resistance_idx = argrelextrema(close_np, np.greater_equal, order=window)[0]
        support_levels = sorted(set(np.round(close_np[support_idx], 2)))
        resistance_levels = sorted(set(np.round(close_np[resistance_idx], 2)))
        for i, lvl in enumerate(support_levels):
            ax_price.axhline(lvl, color='#2ecc71', linestyle=':', linewidth=1, alpha=0.6, label='Support' if i == 0 else None)
            ax_price.annotate(
                f'{lvl}', xy=(df.index[-1], lvl), xytext=(8, 0), textcoords='offset points',
                color='#2ecc71', fontsize=11, fontweight='bold', va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', ec='#2ecc71', lw=1, alpha=0.7)
            )
        for i, lvl in enumerate(resistance_levels):
            ax_price.axhline(lvl, color='#e67e22', linestyle='--', linewidth=1, alpha=0.6, label='Resistance' if i == 0 else None)
            ax_price.annotate(
                f'{lvl}', xy=(df.index[-1], lvl), xytext=(8, 0), textcoords='offset points',
                color='#e67e22', fontsize=11, fontweight='bold', va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', ec='#e67e22', lw=1, alpha=0.7)
            )
        ax_price.set_ylabel('Price', fontsize=15)
        ax_price.legend(loc='upper left', fontsize=12, ncol=2, frameon=True)
        ax_price.set_title(f'{symbol} Price, MA, BB, EMA', fontsize=18)
        ax_price.grid(True, alpha=0.18)

    def plot_price_with_indicators(self, df, symbol: str):
        try:
            import os
            import matplotlib.dates as mdates
            import numpy as np
            from config.config_loader import ConfigLoader
            config_loader = ConfigLoader()
            start_date = config_loader.get('START_DATE', '2025-01-01')
            start_of_period = pd.to_datetime(start_date)
            chart_dpi = int(config_loader.get('CHART_DPI', 200))
            if not df.empty and hasattr(df.index, 'max'):
                df = df[df.index >= start_of_period]
            # Set up subplots: price+MA+BB, RSI, MACD, Stoch
            fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True, gridspec_kw={'height_ratios': [2.5, 1, 1, 1]})
            ax_price, ax_rsi, ax_macd, ax_stoch = axes
            # --- Price, MA, BB, EMA, S/R ---
            self._plot_price_ma_bb_ema_sr(ax_price, df, symbol)
            # --- RSI ---
            if 'rsi' in df.columns:
                ax_rsi.plot(df.index, df['rsi'], label='RSI', color='#8e24aa', linewidth=1.5)
                ax_rsi.axhline(70, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7, label='Overbought (70)')
                ax_rsi.axhline(30, color='#27ae60', linestyle='--', linewidth=1, alpha=0.7, label='Oversold (30)')
                ax_rsi.set_ylabel('RSI', fontsize=13)
                ax_rsi.legend(loc='upper left', fontsize=11, frameon=True)
                ax_rsi.grid(True, alpha=0.18)
            # --- MACD ---
            ax_atr = None
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                ax_macd.plot(df.index, df['macd'], label='MACD', color='#1976d2', linewidth=1.5)
                ax_macd.plot(df.index, df['macd_signal'], label='Signal', color='#e67e22', linewidth=1.2)
                if 'macd_diff' in df.columns:
                    ax_macd.bar(df.index, df['macd_diff'], label='MACD Diff', color='#b2bec3', alpha=0.5, width=1)
                ax_macd.axhline(0, color='#888', linestyle='--', linewidth=1)
                ax_macd.set_ylabel('MACD', fontsize=13)
                ax_macd.grid(True, alpha=0.18)
                handles, labels = ax_macd.get_legend_handles_labels()
                if handles:
                    ax_macd.legend(loc='upper left', fontsize=11, frameon=True)
            # --- ATR (Average True Range) ---
            if 'atr' in df.columns:
                ax_atr = ax_macd.twinx()
                ax_atr.plot(df.index, df['atr'], color='#c0392b', linewidth=1.5, label='ATR')
                ax_atr.set_ylabel('ATR', fontsize=11, color='#c0392b')
                ax_atr.tick_params(axis='y', labelcolor='#c0392b')
                handles, labels = ax_atr.get_legend_handles_labels()
                if handles:
                    ax_atr.legend(loc='upper right', fontsize=10, frameon=True)
            # --- Stochastic Oscillator ---
            ax_mfi = None
            stoch_k_latest = stoch_d_latest = stoch_j_latest = mfi_latest = None
            if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
                ax_stoch.plot(df.index, df['stoch_k'], label='%K', color='#009688', linewidth=1.3)
                ax_stoch.plot(df.index, df['stoch_d'], label='%D', color='#e67e22', linewidth=1.3)
                if 'stoch_j' in df.columns:
                    ax_stoch.plot(df.index, df['stoch_j'], label='%J', color='#e74c3c', linewidth=1.1, linestyle=':')
                ax_stoch.axhspan(80, 100, color='#e74c3c', alpha=0.10, zorder=0)
                ax_stoch.axhspan(0, 20, color='#27ae60', alpha=0.10, zorder=0)
                ax_stoch.axhline(80, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7, label='Overbought (80)')
                ax_stoch.axhline(20, color='#27ae60', linestyle='--', linewidth=1, alpha=0.7, label='Oversold (20)')
                ax_stoch.set_ylabel('Stoch', fontsize=13)
                ax_stoch.set_ylim(0, 100)
                stoch_k_latest = df['stoch_k'].iloc[-1]
                stoch_d_latest = df['stoch_d'].iloc[-1]
                if 'stoch_j' in df.columns:
                    stoch_j_latest = df['stoch_j'].iloc[-1]
                self._style_axis(ax_stoch, 'Stoch')
                self._style_legend(ax_stoch)
                # --- MFI (Money Flow Index) ---
                if 'mfi' in df.columns:
                    ax_mfi = ax_stoch.twinx()
                    ax_mfi.plot(df.index, df['mfi'], color='#d35400', linewidth=1.5, label='MFI')
                    ax_mfi.set_ylim(0, 100)
                    mfi_latest = df['mfi'].iloc[-1]
                    self._style_axis(ax_mfi, 'MFI')
                    self._style_legend(ax_mfi, loc='upper right')
            # --- Volume subplot ---
            if 'volume' in df.columns:
                import matplotlib.gridspec as gridspec
                plt.close(fig)
                fig = plt.figure(figsize=(18, 18))
                gs = gridspec.GridSpec(5, 1, height_ratios=[2.5, 1, 1, 1, 0.7])
                ax_price = fig.add_subplot(gs[0])
                ax_rsi = fig.add_subplot(gs[1], sharex=ax_price)
                ax_macd = fig.add_subplot(gs[2], sharex=ax_price)
                ax_stoch = fig.add_subplot(gs[3], sharex=ax_price)
                ax_vol = fig.add_subplot(gs[4], sharex=ax_price)
                # Re-plot all previous content using helpers
                self._plot_price_ma_bb_ema_sr(ax_price, df, symbol)
                if 'rsi' in df.columns:
                    ax_rsi.plot(df.index, df['rsi'], label='RSI', color='#8e24aa', linewidth=1.5)
                    ax_rsi.axhline(70, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7, label='Overbought (70)')
                    ax_rsi.axhline(30, color='#27ae60', linestyle='--', linewidth=1, alpha=0.7, label='Oversold (30)')
                    ax_rsi.set_ylabel('RSI', fontsize=13)
                    ax_rsi.legend(loc='upper left', fontsize=11, frameon=True)
                    ax_rsi.grid(True, alpha=0.18)
                ax_atr = None
                if 'macd' in df.columns and 'macd_signal' in df.columns:
                    ax_macd.plot(df.index, df['macd'], label='MACD', color='#1976d2', linewidth=1.5)
                    ax_macd.plot(df.index, df['macd_signal'], label='Signal', color='#e67e22', linewidth=1.2)
                    if 'macd_diff' in df.columns:
                        ax_macd.bar(df.index, df['macd_diff'], label='MACD Diff', color='#b2bec3', alpha=0.5, width=1)
                    ax_macd.axhline(0, color='#888', linestyle='--', linewidth=1)
                    ax_macd.set_ylabel('MACD', fontsize=13)
                    ax_macd.grid(True, alpha=0.18)
                    handles, labels = ax_macd.get_legend_handles_labels()
                    if handles:
                        ax_macd.legend(loc='upper left', fontsize=11, frameon=True)
                if 'atr' in df.columns:
                    ax_atr = ax_macd.twinx()
                    ax_atr.plot(df.index, df['atr'], color='#c0392b', linewidth=1.5, label='ATR')
                    ax_atr.set_ylabel('ATR', fontsize=11, color='#c0392b')
                    ax_atr.tick_params(axis='y', labelcolor='#c0392b')
                    handles, labels = ax_atr.get_legend_handles_labels()
                    if handles:
                        ax_atr.legend(loc='upper right', fontsize=10, frameon=True)
                ax_mfi = None
                stoch_k_latest = stoch_d_latest = stoch_j_latest = mfi_latest = None
                if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
                    ax_stoch.plot(df.index, df['stoch_k'], label='%K', color='#009688', linewidth=1.3)
                    ax_stoch.plot(df.index, df['stoch_d'], label='%D', color='#e67e22', linewidth=1.3)
                    if 'stoch_j' in df.columns:
                        ax_stoch.plot(df.index, df['stoch_j'], label='%J', color='#e74c3c', linewidth=1.1, linestyle=':')
                    ax_stoch.axhspan(80, 100, color='#e74c3c', alpha=0.10, zorder=0)
                    ax_stoch.axhspan(0, 20, color='#27ae60', alpha=0.10, zorder=0)
                    ax_stoch.axhline(80, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7, label='Overbought (80)')
                    ax_stoch.axhline(20, color='#27ae60', linestyle='--', linewidth=1, alpha=0.7, label='Oversold (20)')
                    ax_stoch.set_ylabel('Stoch', fontsize=13)
                    ax_stoch.set_ylim(0, 100)
                    stoch_k_latest = df['stoch_k'].iloc[-1]
                    stoch_d_latest = df['stoch_d'].iloc[-1]
                    if 'stoch_j' in df.columns:
                        stoch_j_latest = df['stoch_j'].iloc[-1]
                    self._style_axis(ax_stoch, 'Stoch')
                    self._style_legend(ax_stoch)
                    if 'mfi' in df.columns:
                        ax_mfi = ax_stoch.twinx()
                        ax_mfi.plot(df.index, df['mfi'], color='#d35400', linewidth=1.5, label='MFI')
                        ax_mfi.set_ylim(0, 100)
                        mfi_latest = df['mfi'].iloc[-1]
                        self._style_axis(ax_mfi, 'MFI')
                        self._style_legend(ax_mfi, loc='upper right')
                # Volume
                vol_colors = np.where(df['close'] >= df['close'].shift(1), '#27ae60', '#e74c3c')
                ax_vol.bar(df.index, df['volume'], color=vol_colors, alpha=0.6, label='Volume')
                self._style_axis(ax_vol, 'Volume')
                self._style_legend(ax_vol)
                ax_vol.xaxis.set_major_locator(mdates.MonthLocator())
                ax_vol.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=12, color='#222')
            # --- Chart subtitle and background ---
            date_min = df.index.min().strftime('%Y-%m-%d') if not df.empty else ''
            date_max = df.index.max().strftime('%Y-%m-%d') if not df.empty else ''
            fig.suptitle(f"{symbol} Technical Analysis\n{date_min} to {date_max}", fontsize=22, color='#222', fontweight='bold', y=0.995)
            fig.patch.set_facecolor('#fff')
            for ax in fig.get_axes():
                ax.set_facecolor('#fff')
            # --- Summary box for latest values (Stoch/MFI) ---
            self._draw_summary_box(fig, stoch_k_latest, stoch_d_latest, stoch_j_latest, mfi_latest)
            # --- Mask/fade missing indicator regions ---
            if 'stoch_k' in df.columns:
                mask = df['stoch_k'].isna()
                if mask.any():
                    ax_stoch.fill_between(df.index, 0, 100, where=mask, color='#bbb', alpha=0.15, step='mid')
            if 'mfi' in df.columns and ax_mfi is not None:
                mask = df['mfi'].isna()
                if mask.any():
                    ax_mfi.fill_between(df.index, 0, 100, where=mask, color='#bbb', alpha=0.15, step='mid')
            # --- Save the figure ---
            os.makedirs('stock_indicators', exist_ok=True)
            out_path = f'stock_indicators/{symbol}_chart_detailed.png'
            out_path_svg = f'stock_indicators/{symbol}_chart_detailed.svg'
            plt.savefig(out_path, dpi=chart_dpi, bbox_inches='tight')
            plt.savefig(out_path_svg, format='svg', bbox_inches='tight')
            plt.close()
            self.logger.info(f'Detailed indicator chart saved: {out_path} and {out_path_svg}')
        except Exception as e:
            self.logger.error(f'Error generating detailed indicator chart: {e}')
