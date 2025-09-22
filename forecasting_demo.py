
#!/usr/bin/env python3
"""
forecasting_demo.py

Run this script to generate forecasts on the supplied sample_data.csv.
Outputs:
 - forecast_results.csv (contains actuals and model forecasts)
 - simple console printout of RMSE / MAPE for models
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

DATA_FILE = "sample_data.csv"
OUT_FILE = "forecast_results.csv"

def load_data(path=DATA_FILE):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    return df

def create_features(df, lags=(1,7,30)):
    data = df.copy()
    for lag in lags:
        data[f"lag_{lag}"] = data["sales"].shift(lag)
    data["rolling_mean_7"] = data["sales"].shift(1).rolling(7).mean()
    data["dayofweek"] = data.index.dayofweek
    data["month"] = data.index.month
    data = data.dropna()
    return data

def train_test_split(data, train_frac=0.8):
    n = int(len(data) * train_frac)
    train = data.iloc[:n].copy()
    test = data.iloc[n:].copy()
    return train, test

def evaluate(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mape

def baseline_naive(test):
    # naïve: next-step equals previous day (lag_1)
    return test["lag_1"]

def try_sarima(train, test):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as e:
        print("statsmodels is not available or failed to import SARIMAX:", e)
        return None
    # simple SARIMA with seasonal weekly component
    model = SARIMAX(train["sales"], order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.predict(start=test.index[0], end=test.index[-1])
    return pred

def try_xgboost(train, test, features):
    try:
        import xgboost as xgb
    except Exception as e:
        print("xgboost not available:", e)
        return None
    X_train = train[features]
    y_train = train["sales"]
    X_test = test[features]
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return pd.Series(pred, index=test.index)

def main():
    print("Loading data...")
    df = load_data(DATA_FILE)
    print(f"Loaded {len(df)} rows. Date range: {df.index.min().date()} to {df.index.max().date()}")

    data = create_features(df)
    train, test = train_test_split(data, train_frac=0.8)
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    results = test[["sales"]].copy()

    # Baseline
    results["naive"] = baseline_naive(test)
    rmse_naive, mape_naive = evaluate(results["sales"], results["naive"])
    print(f"Naive -> RMSE: {rmse_naive:.2f}, MAPE: {mape_naive:.2f}%")

    # SARIMA
    sarima_pred = try_sarima(train, test)
    if sarima_pred is not None:
        sarima_pred = sarima_pred.reindex(test.index)  # ensure matching index
        results["sarima"] = sarima_pred
        rmse_sarima, mape_sarima = evaluate(results["sales"], results["sarima"])
        print(f"SARIMA -> RMSE: {rmse_sarima:.2f}, MAPE: {mape_sarima:.2f}%")
    else:
        print("SARIMA skipped due to import error.")

    # XGBoost
    features = [c for c in data.columns if c not in ["sales"]]
    xgb_pred = try_xgboost(train, test, features)
    if xgb_pred is not None:
        results["xgboost"] = xgb_pred
        rmse_xgb, mape_xgb = evaluate(results["sales"], results["xgboost"])
        print(f"XGBoost -> RMSE: {rmse_xgb:.2f}, MAPE: {mape_xgb:.2f}%")
    else:
        print("XGBoost skipped due to import error.")

    # Save results
    results.to_csv(OUT_FILE)
    print(f"Forecast results written to {OUT_FILE} in current folder.")

if __name__ == '__main__':
    main()
