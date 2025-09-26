#!/usr/bin/env python3
"""
forecasting_demo_fixed.py

Complete program: original demo with cross-version sklearn compatibility and robustness fixes.
- Safe evaluate() (RMSE compatible with older sklearn; MAPE robust to zeros)
- Index alignment for predictions/actuals
- SARIMA / XGBoost imports handled gracefully
- Predictions reindexed to test.index
- Saves forecast_results.csv
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


def create_features(df, lags=(1, 7, 30)):
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
    """
    Compute RMSE and MAPE (robust to zero actuals).
    Works across sklearn versions (does not rely on squared=False).
    Aligns indices before computing metrics.
    Returns (rmse, mape_percent)
    """
    # align indices (inner intersection)
    y_true, y_pred = y_true.align(y_pred, join="inner")

    # RMSE: some sklearn versions don't accept squared=False; take sqrt of MSE instead
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    # MAPE: ignore zero actuals to avoid divide-by-zero
    mask = (y_true != 0)
    if mask.sum() == 0:
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

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
    try:
        # simple SARIMA with seasonal weekly component
        model = SARIMAX(
            train["sales"],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        # get_prediction ensures predictable results and index compatibility
        pred = res.get_prediction(start=test.index[0], end=test.index[-1]).predicted_mean
        pred = pred.reindex(test.index)  # ensure matching index (fills NaN if necessary)
        return pred
    except Exception as e:
        print("SARIMA training/prediction failed:", e)
        return None


def try_xgboost(train, test, features):
    try:
        import xgboost as xgb
    except Exception as e:
        print("xgboost not available:", e)
        return None
    X_train = train[features]
    y_train = train["sales"]
    X_test = test[features]

    # Ensure no NaNs in features (drop any rows with NaNs if present)
    train_mask = X_train.notna().all(axis=1)
    test_mask = X_test.notna().all(axis=1)
    if not train_mask.all():
        print(f"Warning: dropping { (~train_mask).sum() } training rows due to NaNs in features.")
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
    if not test_mask.all():
        print(f"Warning: dropping { (~test_mask).sum() } test rows due to NaNs in features.")
        X_test = X_test[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        print("Not enough data for XGBoost after dropping NaNs.")
        return None

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return pd.Series(pred, index=X_test.index)


def main():
    print("Loading data...")
    df = load_data(DATA_FILE)
    print(f"Loaded {len(df)} rows. Date range: {df.index.min().date()} to {df.index.max().date()}")

    data = create_features(df)
    train, test = train_test_split(data, train_frac=0.8)
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    results = test[["sales"]].copy()

    # Baseline
    try:
        results["naive"] = baseline_naive(test)
        rmse_naive, mape_naive = evaluate(results["sales"], results["naive"])
        print(f"Naive -> RMSE: {rmse_naive:.2f}, MAPE: {mape_naive:.2f}%")
    except Exception as e:
        print("Naive baseline failed:", e)

    # SARIMA
    sarima_pred = try_sarima(train, test)
    if sarima_pred is not None:
        results["sarima"] = sarima_pred
        try:
            rmse_sarima, mape_sarima = evaluate(results["sales"], results["sarima"])
            print(f"SARIMA -> RMSE: {rmse_sarima:.2f}, MAPE: {mape_sarima:.2f}%")
        except Exception as e:
            print("Error evaluating SARIMA predictions:", e)
    else:
        print("SARIMA skipped due to import or training error.")

    # XGBoost
    features = [c for c in data.columns if c not in ["sales"]]
    xgb_pred = try_xgboost(train, test, features)
    if xgb_pred is not None:
        results["xgboost"] = xgb_pred.reindex(results.index)
        try:
            rmse_xgb, mape_xgb = evaluate(results["sales"], results["xgboost"])
            print(f"XGBoost -> RMSE: {rmse_xgb:.2f}, MAPE: {mape_xgb:.2f}%")
        except Exception as e:
            print("Error evaluating XGBoost predictions:", e)
    else:
        print("XGBoost skipped due to import error.")

    # Save results
    results.to_csv(OUT_FILE)
    print(f"Forecast results written to {OUT_FILE} in current folder.")


if __name__ == '__main__':
    main()
