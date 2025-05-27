"""
Recommendation module for Stock Analysis Tool.
Generates investment recommendations based on analysis and user profile.
"""
import pandas as pd
import os
from dotenv import load_dotenv
from logs.logger_setup import LoggerSetup
import json as _json
from typing import Optional

class RecommendationEngine:
    """
    Provides investment recommendations and position management guidelines.
    """
    def __init__(self, user_profile: Optional[dict] = None, log_level: str = 'INFO'):
        """
        Initialize the RecommendationEngine.
        Args:
            user_profile (dict, optional): User profile for personalized recommendations.
            log_level (str): Logging level.
        """
        self.user_profile = user_profile or {}
        self.logger = LoggerSetup(log_level=log_level).get_logger()

    def _extract_indicator_context(self, df: pd.DataFrame, idx) -> tuple:
        """
        Extracts indicator values from df for a given index.
        Args:
            df (pd.DataFrame): DataFrame with indicator columns.
            idx: Index to extract values for.
        Returns:
            Tuple of (price, rsi, macd, macd_signal, sma_20, sma_50, volatility)
        """
        if idx in df.index:
            price = df.loc[idx, 'close']
            rsi = df.loc[idx, 'rsi'] if 'rsi' in df.columns else None
            macd = df.loc[idx, 'macd'] if 'macd' in df.columns else None
            macd_signal = df.loc[idx, 'macd_signal'] if 'macd_signal' in df.columns else None
            sma_20 = df.loc[idx, 'sma_20'] if 'sma_20' in df.columns else None
            sma_50 = df.loc[idx, 'sma_50'] if 'sma_50' in df.columns else None
            volatility = df.loc[idx, 'volatility'] if 'volatility' in df.columns else None
        else:
            price = rsi = macd = macd_signal = sma_20 = sma_50 = volatility = None
        return price, rsi, macd, macd_signal, sma_20, sma_50, volatility

    def _get_trend(self, sma_20, sma_50) -> Optional[str]:
        """
        Returns trend label based on SMA20 and SMA50.
        Args:
            sma_20: SMA 20 value.
            sma_50: SMA 50 value.
        Returns:
            Trend label as 'Uptrend', 'Downtrend', or 'Sideways'; None if unable to determine.
        """
        if sma_20 and sma_50:
            if sma_20 > sma_50:
                return 'Uptrend'
            elif sma_20 < sma_50:
                return 'Downtrend'
            else:
                return 'Sideways'
        return None

    def _get_risk(self, volatility) -> Optional[str]:
        """
        Returns risk level based on volatility value.
        Args:
            volatility: Volatility value.
        Returns:
            Risk level as 'High', 'Medium', or 'Low'; None if unable to determine.
        """
        if volatility:
            if volatility > 0.35:
                return 'High'
            elif volatility > 0.2:
                return 'Medium'
            else:
                return 'Low'
        return None

    def _build_rationale(self, row, price, rsi, macd, macd_signal, sma_20, sma_50, trend, volatility, risk) -> str:
        """
        Builds a rationale string for the recommendation.
        Args:
            row: Signal row from the DataFrame.
            price: Current price.
            rsi: RSI value.
            macd: MACD value.
            macd_signal: MACD signal value.
            sma_20: SMA 20 value.
            sma_50: SMA 50 value.
            trend: Trend label.
            volatility: Volatility value.
            risk: Risk level.
        Returns:
            Rationale string for the recommendation.
        """
        return (
            f"Signal: {row['signal'].capitalize()} | Confidence: {row['confidence']} | Price: {price} | "
            f"RSI: {rsi} | MACD: {macd} | MACD Signal: {macd_signal} | SMA20: {sma_20} | "
            f"SMA50: {sma_50} | Trend: {trend} | Volatility: {volatility:.2%} | Risk: {risk}. "
            f"Reason: {row['reason']}"
        )

    def generate_recommendations(self, df: pd.DataFrame, signals: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Generates personalized investment recommendations and saves them to CSV and JSON files.
        Args:
            df (pd.DataFrame): Stock data with indicators.
            signals (pd.DataFrame): Signal DataFrame with 'signal', 'confidence', and 'reason'.
            symbol (str, optional): Stock symbol for output file naming.
        Returns:
            pd.DataFrame: DataFrame of recommendations.
        """
        if df.empty or signals.empty:
            self.logger.error("No data or signals for recommendations.")
            return pd.DataFrame()
        try:
            recs = []
            trade_log_path = f'backtesting_results/{symbol}_trade_log.csv' if symbol else None
            trade_log = None
            if trade_log_path and os.path.exists(trade_log_path):
                try:
                    trade_log = pd.read_csv(trade_log_path, parse_dates=['open_date'])
                    trade_log.set_index('open_date', inplace=True)
                    self.logger.info(f"Loaded trade log from {trade_log_path}.")
                except Exception as e:
                    self.logger.warning(f"Could not load trade log for {symbol}: {e}")
            df['volatility'] = df['close'].pct_change().rolling(10).std() * (252 ** 0.5)
            for idx, row in signals.iterrows():
                price, rsi, macd, macd_signal, sma_20, sma_50, volatility = self._extract_indicator_context(df, idx)
                trend = self._get_trend(sma_20, sma_50)
                risk = self._get_risk(volatility)
                rationale = self._build_rationale(row, price, rsi, macd, macd_signal, sma_20, sma_50, trend, volatility, risk)
                pnl = trade_log.loc[idx, 'pnl'] if trade_log is not None and idx in trade_log.index else None
                expected_return = pnl if pnl is not None else (0.05 if row['signal']=='buy' else (-0.05 if row['signal']=='sell' else 0))
                position_size = min(1.0, max(0.2, row['confidence']/100)) if row['signal'] in ['buy','sell'] else 0
                recommended_price = price if row['signal'] in ['buy','sell'] else None
                # --- Add best buy/sell price ---
                best_buy_price = None
                best_sell_price = None
                if trade_log is not None:
                    if row['signal'] == 'buy':
                        # Find the most recent buy trade before or at idx
                        prev_buys = trade_log[(trade_log.index <= idx) & (trade_log['entry_type'].str.lower().str.contains('buy'))]
                        if not prev_buys.empty:
                            best_buy_price = prev_buys.iloc[-1]['open_price']
                    if row['signal'] == 'sell':
                        # Find the most recent sell trade before or at idx
                        prev_sells = trade_log[(trade_log.index <= idx) & (trade_log['exit_type'].str.lower().str.contains('sell'))]
                        if not prev_sells.empty:
                            best_sell_price = prev_sells.iloc[-1]['close_price']
                if best_buy_price is None and row['signal'] == 'buy':
                    best_buy_price = price
                if best_sell_price is None and row['signal'] == 'sell':
                    best_sell_price = price
                rec = {
                    'date': idx,
                    'recommendation': row['signal'].capitalize(),
                    'confidence': row['confidence'],
                    'risk': risk,
                    'expected_return': expected_return,
                    'position_size': position_size,
                    'trend': trend,
                    'volatility': volatility,
                    'price': price,
                    'recommended_price': recommended_price,
                    'best_buy_price': best_buy_price,
                    'best_sell_price': best_sell_price,
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'rationale': rationale,
                    'reason': row['reason'] if 'reason' in row else '',
                }
                for col in signals.columns:
                    if col not in rec and col in row:
                        rec[col] = row[col]
                recs.append(rec)
            load_dotenv()
            start_date = os.getenv('START_DATE', '2020-01-01')
            rec_df = pd.DataFrame(recs)
            rec_df['date'] = pd.to_datetime(rec_df['date'])
            rec_df = rec_df[rec_df['date'] >= pd.to_datetime(start_date)]
            # Ensure 'reason' is always a JSON string for output
            if 'reason' in rec_df.columns:
                rec_df['reason'] = rec_df['reason'].apply(lambda x: _json.dumps(x) if isinstance(x, dict) else (x if isinstance(x, str) else str(x)))
            if symbol:
                os.makedirs('stock_recommendations', exist_ok=True)
                columns_order = [
                    'date','recommendation','confidence','risk','expected_return','position_size','trend','volatility','price','recommended_price','best_buy_price','best_sell_price','rsi','macd','macd_signal','sma_20','sma_50','rationale','reason'
                ]
                # Add any extra columns at the end
                all_cols = list(rec_df.columns)
                ordered_cols = [c for c in columns_order if c in all_cols] + [c for c in all_cols if c not in columns_order]
                csv_path = f'stock_recommendations/{symbol}_recommendations.csv'
                json_path = f'stock_recommendations/{symbol}_recommendations.json'
                try:
                    if not rec_df.empty:
                        rec_df.to_csv(csv_path, index=False, columns=ordered_cols)
                        self.logger.info(f"Saved {len(rec_df)} recommendations to CSV: {csv_path}")
                except Exception as e:
                    self.logger.error(f"Error saving recommendations CSV to {csv_path}: {e}")
                # Also save as pretty-printed JSON
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        # Convert all date columns to string for JSON serialization
                        if 'date' in rec_df.columns:
                            rec_df['date'] = rec_df['date'].apply(lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x) if pd.notnull(x) else '')
                        _json.dump(rec_df[ordered_cols].to_dict(orient='records'), f, indent=2, ensure_ascii=False)
                        self.logger.info(f"Saved recommendations to JSON: {json_path}")
                except Exception as e:
                    self.logger.error(f"Error saving recommendations JSON to {json_path}: {e}")
            self.logger.info(f"Recommendations generated and saved for {symbol} (total: {len(rec_df)})")
            return rec_df
        except Exception as e:
            self.logger.error(f"Error generating recommendations for {symbol if symbol else ''}: {e}", exc_info=True)
            return pd.DataFrame()
