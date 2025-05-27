"""
Forecast Engine for Stock Analysis Tool
Implements time series forecasting for stock prices and indicators using statistical and machine learning models.
"""
import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from logs.logger_setup import LoggerSetup

# Optional: import models if available
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

class ForecastEngine:
    """
    Provides time series forecasting for stock prices and indicators.
    Supports ARIMA and Prophet (if installed), extensible for ML/AI models.
    """
    def __init__(self, log_level: str = 'INFO'):
        """
        Initialize the ForecastEngine.
        Args:
            log_level (str): Logging level.
        """
        self.logger = LoggerSetup(log_level=log_level).get_logger()
        load_dotenv()
        self.forecast_dir = os.getenv('FORECAST_DIR', 'stock_forecast')
        os.makedirs(self.forecast_dir, exist_ok=True)
        # Model registry for extensibility
        self.model_registry = {
            'arima': self.forecast_arima,
            #'prophet': self.forecast_prophet
            # Future: add ML/ensemble models here
        }

    def forecast_arima(self, df: pd.DataFrame, column: str, order: tuple = (5,1,0), steps: int = 5) -> Optional[pd.DataFrame]:
        """
        Forecast using ARIMA model.
        Args:
            df (pd.DataFrame): Input data.
            column (str): Column to forecast.
            order (tuple): ARIMA order.
            steps (int): Forecast horizon.
        Returns:
            pd.DataFrame or None: Forecast results.
        """
        if ARIMA is None:
            self.logger.error("statsmodels is not installed. Cannot run ARIMA.")
            return None
        if column not in df.columns:
            self.logger.error(f"Column '{column}' not in DataFrame.")
            return None
        series = df[column].dropna()
        # Fine-tuned frequency handling for ARIMA
        if hasattr(series.index, 'freq') and series.index.freq is None:
            inferred = pd.infer_freq(series.index)
            if inferred:
                try:
                    series.index = pd.DatetimeIndex(series.index, freq=inferred)
                except Exception as e:
                    self.logger.warning(f"Could not set inferred frequency '{inferred}' on index: {e}")
            else:
                # Only resample if index is monotonic, unique, has at least 2 points, and not already business day freq
                if (series.index.is_monotonic_increasing and series.index.is_unique and len(series) > 1 and not pd.infer_freq(series.index) == 'B'):
                    try:
                        series = series.asfreq('B')
                        self.logger.warning("Could not infer frequency for date index; resampled to business day frequency (B) for ARIMA.")
                    except Exception as e:
                        self.logger.warning(f"Could not infer or resample frequency for date index; ARIMA may issue a ValueWarning. Error: {e}")
                else:
                    self.logger.warning("Date index is not monotonic, unique, too short, or already business day freq; proceeding with original index for ARIMA.")
        try:
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            forecast = model_fit.get_forecast(steps=steps)
            forecast_df = forecast.summary_frame()
            forecast_df.index.name = 'date'
            self.logger.info(f"ARIMA forecast complete for '{column}'.")
            return forecast_df
        except Exception as e:
            self.logger.error(f"ARIMA forecast failed for '{column}': {e}")
            return None

    def forecast_prophet(self, df: pd.DataFrame, column: str, steps: int = 5) -> Optional[pd.DataFrame]:
        """
        Forecast using Prophet model.
        Args:
            df (pd.DataFrame): Input data.
            column (str): Column to forecast.
            steps (int): Forecast horizon.
        Returns:
            pd.DataFrame or None: Forecast results.
        """
        if Prophet is None:
            self.logger.error("Prophet is not installed. Cannot run Prophet forecast.")
            return None
        if column not in df.columns:
            self.logger.error(f"Column '{column}' not in DataFrame.")
            return None
        try:
            prophet_df = df[[column]].reset_index().rename(columns={'index': 'ds', column: 'y'})
            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=steps)
            forecast = m.predict(future)
            self.logger.info(f"Prophet forecast complete for '{column}'.")
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')
        except Exception as e:
            self.logger.error(f"Prophet forecast failed for '{column}': {e}")
            return None

    def save_forecast(self, forecast_df: Optional[pd.DataFrame], symbol: str, model: str, column: str) -> None:
        """
        Save forecast results to CSV and JSON.
        Args:
            forecast_df (pd.DataFrame): Forecast results.
            symbol (str): Stock symbol.
            model (str): Model name.
            column (str): Forecasted column.
        """
        if forecast_df is None or forecast_df.empty:
            self.logger.warning(f"No forecast to save for {symbol} {model} {column}.")
            return
        csv_path = os.path.join(self.forecast_dir, f"{symbol}_{model}_{column}_forecast.csv")
        json_path = os.path.join(self.forecast_dir, f"{symbol}_{model}_{column}_forecast.json")
        try:
            forecast_df.to_csv(csv_path, index=True, date_format='%Y-%m-%d')
            forecast_df.reset_index().to_json(json_path, orient='records', date_format='iso', indent=2)
            self.logger.info(f"Saved forecast to {csv_path} and {json_path}.")
        except Exception as e:
            self.logger.error(f"Error saving forecast for {symbol} {model} {column}: {e}")

    def run(self, df: pd.DataFrame, symbol: str, column: str = 'close', steps: int = 10, **kwargs) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Run forecast for a given symbol and column using all registered models.
        Args:
            df (pd.DataFrame): Input data with a datetime index.
            symbol (str): Stock symbol.
            column (str): Column to forecast (default 'close').
            steps (int): Forecast horizon.
        Returns:
            Dict[str, pd.DataFrame or None]: Forecast results for each model.
        """
        results = {}
        for model_name, model_func in self.model_registry.items():
            forecast_df = model_func(df, column, steps=steps, **kwargs)
            self.save_forecast(forecast_df, symbol, model_name, column)
            results[model_name] = forecast_df
        return results
