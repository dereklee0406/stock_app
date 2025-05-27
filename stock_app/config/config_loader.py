"""
Configuration loader for Stock Analysis Tool.
Loads environment variables and config files.
"""
import os
from dotenv import load_dotenv

class ConfigLoader:
    """
    Loads configuration from .env and environment variables.
    """
    def __init__(self, env_path: str = '.env'):
        load_dotenv(env_path)

    def get(self, key: str, default=None):
        return os.getenv(key, default)

    def get_float(self, key: str, default=None):
        val = os.getenv(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def get_all(self):
        """
        Returns all loaded environment variables as a dictionary.
        """
        return dict(os.environ)
