"""
Data acquisition module for Stock Analysis Tool.
Handles retrieval and caching of stock data from Alpha Vantage.
"""
import os
import pandas as pd
import requests
import time
from dotenv import load_dotenv
from config.config_loader import ConfigLoader
from logs.logger_setup import LoggerSetup

class DataAcquisition:
    """
    Handles data retrieval, caching, and validation for stock data.
    """
    def __init__(self, api_key: str = None, cache_dir: str = None, log_level: str = 'INFO'):
        load_dotenv()
        config = ConfigLoader()
        self.api_key = api_key or config.get('ALPHA_VANTAGE_API_KEY')
        self.cache_dir = cache_dir or config.get('CACHE_DIR', 'stock_data')
        self.default_interval = config.get('DEFAULT_INTERVAL', 'daily')
        self.logger = LoggerSetup(log_level=config.get('LOG_LEVEL', log_level)).get_logger()
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_data(self, symbol: str, interval: str = 'daily') -> pd.DataFrame:
        """
        Fetches historical stock data from Alpha Vantage free API or cache.
        Always fetches full history (outputsize='full') for all intervals.
        Only free endpoints are used: TIME_SERIES_DAILY, TIME_SERIES_WEEKLY, TIME_SERIES_MONTHLY, TIME_SERIES_INTRADAY.
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_data_{interval}.csv")
        now = time.time()
        cache_validity = 3600 if interval in ['1min', '5min', '15min', '30min', '60min'] else 86400
        if os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            if now - mtime < cache_validity:
                self.logger.info(f"Loading cached data for {symbol} ({interval})")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        self.logger.info(f"Fetching data for {symbol} ({interval}) from Alpha Vantage API (free tier)")
        url, params = self._build_api_request(symbol, interval)
        params['outputsize'] = 'full'  # Always get full history
        for attempt in range(5):
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'Error Message' in data or 'Note' in data:
                    self.logger.warning(f"API error: {data.get('Error Message', data.get('Note', 'Unknown error'))}")
                    self.logger.error(f"Full API response: {data}")
                    if 'Note' in data and 'frequency' in data['Note']:
                        time.sleep(2 ** attempt)
                        continue
                    return pd.DataFrame()
                df = self._parse_response(data, interval)
                if df.empty:
                    self.logger.error(f"No time series data found. Full API response: {data}")
                if not df.empty:
                    df.to_csv(cache_file)
                return df
            elif response.status_code == 429:
                self.logger.warning("API rate limit hit, retrying...")
                time.sleep(2 ** attempt)
            else:
                self.logger.error(f"HTTP error {response.status_code}")
                break
        return pd.DataFrame()

    def _build_api_request(self, symbol, interval):
        # Always use free Alpha Vantage endpoints (TIME_SERIES_*)
        if interval in ['1min', '5min', '15min', '30min', '60min']:
            function = 'TIME_SERIES_INTRADAY'
        elif interval == 'daily':
            function = 'TIME_SERIES_DAILY'
        elif interval == 'weekly':
            function = 'TIME_SERIES_WEEKLY'
        elif interval == 'monthly':
            function = 'TIME_SERIES_MONTHLY'
        else:
            raise ValueError(f"Unsupported interval: {interval}")
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'datatype': 'json',
        }
        if function == 'TIME_SERIES_INTRADAY':
            params['interval'] = interval
        return url, params

    def _parse_response(self, data, interval):
        # Find the correct key for time series
        for key in data:
            if 'Time Series' in key:
                ts_key = key
                break
        else:
            self.logger.error("No time series data found in API response.")
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(data[ts_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        # Standardize column names
        col_map = {
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume',
        }
        df = df.rename(columns=col_map)
        # Keep only standard OHLCV columns
        keep_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in keep_cols:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[keep_cols]
        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validates the completeness and plausibility of the data.
        Only warns if required columns (open, high, low, close, volume) are all NaN.
        """
        if df.empty:
            self.logger.error("DataFrame is empty.")
            return False
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        # Ensure numeric columns for comparison
        for col in required_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Only warn if required columns are all NaN
        for col in required_cols:
            if col in df.columns and df[col].isnull().all():
                self.logger.warning(f"Column '{col}' is all NaN.")
        if not all(col in df.columns for col in required_cols):
            self.logger.error("Missing required OHLCV columns.")
            return False
        if (df[required_cols] < 0).any().any():
            self.logger.warning("Negative values found in OHLCV columns.")
        return True
