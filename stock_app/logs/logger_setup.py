"""
Logging setup for Stock Analysis Tool.
Configures multi-level logging to file and console.
"""
import logging
import os

class LoggerSetup:
    """
    Sets up logging for the application.
    """
    def __init__(self, log_dir: str = 'logs', log_level: str = 'INFO'):
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'stock_app.log')
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format='%(asctime)s [%(levelname)s] %(module)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('StockApp')

    def get_logger(self):
        return self.logger
