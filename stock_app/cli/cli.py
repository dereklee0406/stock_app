"""
CLI module for Stock Analysis Tool.
Handles command-line interface and user interaction.
"""
import argparse
import os
import time
from data_acquisition.data_acquisition import DataAcquisition
from technical_analysis.technical_analysis import TechnicalAnalysis, ChartGenerator
from signal_generation.signal_generator import SignalGenerator, plot_signals
from strategy.strategy import Strategy
from recommendation.recommendation_engine import RecommendationEngine
from forecast.forecast_engine import ForecastEngine

class StockAppCLI:
    """
    Command-line interface for the Stock Analysis Tool.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Stock Analysis Tool CLI')
        self.parser.add_argument('--watchlist', type=str, default='watchlist.txt', help='Path to file with list of symbols (one per line)')
        self.parser.add_argument('--interval', type=str, default=None, help='Time interval (e.g., daily, 1min)')
        self.parser.add_argument('--force-refresh', action='store_true', help='Force refresh data from API')
        self.parser.add_argument('--use-cache', action='store_true', help='Use cached data if available, do not fetch from API')
        # Forecast arguments (model removed, always run all models)
        self.parser.add_argument('--forecast', action='store_true', help='Run forecast (all models) for all symbols in watchlist')
        self.parser.add_argument('--forecast-column', type=str, default='close', help='Column to forecast (default: close)')
        self.parser.add_argument('--forecast-steps', type=int, default=365, help='Forecast horizon (steps)')

    def run(self):
        args = self.parser.parse_args()
        symbols = []
        watchlist_path = args.watchlist if hasattr(args, 'watchlist') and args.watchlist else 'watchlist.txt'
        if os.path.exists(watchlist_path):
            with open(watchlist_path) as f:
                symbols = [line.strip() for line in f if line.strip()]
        if not symbols:
            print('No symbols provided in watchlist.')
            return
        import concurrent.futures
        def process_symbol(symbol):
            print(f'Processing {symbol}...')
            da = DataAcquisition()
            interval = args.interval or da.default_interval
            cache_file = os.path.join(da.cache_dir, f"{symbol}_data_{interval}.csv")
            use_cache = False
            if os.path.exists(cache_file):
                mtime = os.path.getmtime(cache_file)
                age = time.time() - mtime
                if age < 86400 and not args.force_refresh:
                    use_cache = True
            if use_cache:
                df = da.fetch_data(symbol, interval=interval)
            else:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                df = da.fetch_data(symbol, interval=interval)
            if not da.validate_data(df):
                print(f'Invalid data for {symbol}. Skipping.')
                return
            ta = TechnicalAnalysis()
            df = ta.calculate_indicators(df, symbol=symbol)
            # Generate chart after indicators
            chart = ChartGenerator()
            chart.plot_price_with_indicators(df, symbol)
            sg = SignalGenerator()
            df = sg.generate_signals(df, symbol=symbol)
            st = Strategy()
            st.backtest(df, df, symbol=symbol)
            re = RecommendationEngine()
            re.generate_recommendations(df, df, symbol=symbol)
            # Forecast integration: always run all models
            if getattr(args, 'forecast', False):
                fe = ForecastEngine()
                try:
                    fe.run(df, symbol, column=args.forecast_column, steps=args.forecast_steps)
                except Exception as e:
                    print(f'Forecast error for {symbol}: {e}')
            print(f'Completed {symbol}.')
        # Use ThreadPoolExecutor for multi-threaded processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_symbol, symbol) for symbol in symbols]
            for future in concurrent.futures.as_completed(futures):
                # Optionally print exceptions
                exc = future.exception()
                if exc:
                    print(f'Error in thread: {exc}')
