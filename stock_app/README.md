# Stock Analysis Tool

A robust, modular, and object-oriented Python tool for comprehensive stock analysis, technical signal generation, advanced backtesting, and professional investment recommendations using the Alpha Vantage API.

## Features
- Data acquisition and intelligent caching
- Technical analysis with multiple indicators (RSI, MACD, SMA, volatility, etc.)
- Trading signal generation with structured, explainable reasons
- Strategy development and advanced backtesting (robust trade log, summary export, edge-case handling)
- Investment recommendations with rationale and risk assessment
- Professional reporting: outputs in CSV and JSON, ready for downstream analytics
- Batch and multithreaded processing ready
- Robust error handling and logging (see `logs/`)
- Extensible, maintainable, and production-ready codebase
- Command-line interface (CLI)

## Setup
1. Clone the repository.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your Alpha Vantage API key and other settings.
4. Run the CLI:
   ```sh
   python main.py --help
   ```

## Project Structure
- `data_acquisition/` – Download and cache stock data
- `technical_analysis/` – Calculate indicators
- `signal_generation/` – Generate trading signals (with structured `reason` fields)
- `strategy/` – Strategy development and advanced backtesting (robust trade log, summary export)
- `recommendation/` – Investment recommendations (with rationale and risk assessment)
- `cli/` – Command-line interface
- `config/` – Configuration files
- `logs/` – Log files (detailed error and process logs)
- Output directories:
  - `stock_data/` – Raw and processed stock data
  - `stock_indicators/` – Indicator calculations and charts
  - `stock_signal/` – Signal history and visualizations
  - `backtesting_results/` – Trade logs and backtest summaries (CSV/JSON)
  - `stock_recommendations/` – Investment recommendations (CSV/JSON)

## Output & Reporting
- All outputs are robust to edge cases and non-serializable types.
- Trade logs and summaries are exported as structured JSON and CSV, with all `reason` fields as objects (not strings).
- Recommendations and signals are ready for downstream analytics and batch/multithreaded processing.

## Troubleshooting
- Ensure your API key is valid and not rate-limited.
- Check logs in the `logs/` directory for errors and process details.
- For serialization or export issues, verify your Python and package versions match `requirements.txt`.

## License
MIT License
