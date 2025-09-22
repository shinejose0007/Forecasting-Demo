
# Sales Forecasting Demo (Time Series)

This repository is a compact, runnable demo for **time series forecasting** with Python.
It demonstrates an end-to-end workflow useful for job applications and portfolio showcase:
- ETL / data preparation
- Feature engineering for time series (lags, rolling means, calendar features)
- Baselines (naïve), classical (SARIMA/ETS) and ML models (XGBoost)
- Backtesting / time-series split and evaluation (RMSE, MAPE)
- Export of forecast results for Power BI integration

## Files in this package
- `forecasting_demo.py` — runnable Python script that trains models on sample data and exports `forecast_results.csv`.
- `sample_data.csv` — synthetic daily sales data (2020-01-01 to 2022-12-31).
- `requirements.txt` — pip-installable dependencies for the demo.
- `README.md` — this file.

## Quickstart
1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```
2. Run the demo (it will create `forecast_results.csv`):
```bash
python forecasting_demo.py
```
3. Open `forecast_results.csv` or load it in Power BI to build a dashboard.

## Notes
- SARIMA (statsmodels) and XGBoost are included in the notebook/script. If you cannot install these packages, the script prints a helpful message and continues with available models.
- The sample data is synthetic — replace `sample_data.csv` with your own sales data in the same format (`date, sales`) to run real scenarios.
- The script is intentionally simple and structured so you can paste parts into a Jupyter notebook for interactive exploration.
