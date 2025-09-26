#!/usr/bin/env python3
"""
forecasting_demo_with_fallback.py

Behavior:
 - Loads sample_data.csv (expects 'date' and 'sales')
 - Creates lag / rolling features
 - Baseline naive forecast
 - SARIMA if statsmodels is installed
 - XGBoost if available, otherwise falls back to sklearn RandomForestRegressor
 - Safe RMSE/MAPE evaluate() compatible with older sklearn
 - Writes forecast_results.csv
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error

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
    """RMSE and MAPE (robust to zeros), align indices."""
    y_true, y_pred = y_true.align(y_pred, join="inner")
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mask = (y_true != 0)
    if mask.sum() == 0:
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return rmse, mape


def baseline_naive(test):
    return test["lag_1"]


def try_sarima(train, test):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as e:
        print("statsmodels is not available or failed to import SARIMAX:", e)
        return None
    try:
        model = SARIMAX(
            train["sales"],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        pred = res.get_prediction(start=test.index[0], end=test.index[-1]).predicted_mean
        pred = pred.reindex(test.index)
        return pred
    except Exception as e:
        print("SARIMA training/prediction failed:", e)
        return None


def try_xgboost_or_rf(train, test, features, random_state=42):
    """
    Try xgboost; if not present, fall back to sklearn RandomForestRegressor.
    Returns (pred_series, model_name)
    """
    X_train = train[features].copy()
    y_train = train["sales"].copy()
    X_test = test[features].copy()

    # Drop rows with NaNs just in case
    train_mask = X_train.notna().all(axis=1)
    test_mask = X_test.notna().all(axis=1)
    if not train_mask.all():
        X_train = X_train[train_mask]; y_train = y_train[train_mask]
    if not test_mask.all():
        X_test = X_test[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        print("Not enough data for tree-based model after dropping NaNs.")
        return None, None

    # Try xgboost first
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return pd.Series(preds, index=X_test.index), "xgboost"
    except Exception as e_xgb:
        # fallback to RandomForest
        from sklearn.ensemble import RandomForestRegressor
        print("xgboost not available, falling back to RandomForestRegressor. Info:", e_xgb)
        rf = RandomForestRegressor(n_estimators=200, random_state=random_state)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        return pd.Series(preds, index=X_test.index), "random_forest"


def main():
    print("Loading data...")
    df = load_data(DATA_FILE)
    print(f"Loaded {len(df)} rows. Date range: {df.index.min().date()} to {df.index.max().date()}")

    data = create_features(df)
    train, test = train_test_split(data, train_frac=0.8)
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    results = test[["sales"]].copy()

    # Naive baseline
    try:
        results["naive"] = baseline_naive(test)
        rmse_naive, mape_naive = evaluate(results["sales"], results["naive"])
        print(f"Naive -> RMSE: {rmse_naive:.2f}, MAPE: {mape_naive:.2f}%")
    except Exception as e:
        print("Naive baseline failed:", e)

    # SARIMA (optional)
    sarima_pred = try_sarima(train, test)
    if sarima_pred is not None:
        results["sarima"] = sarima_pred
        try:
            rmse_sarima, mape_sarima = evaluate(results["sales"], results["sarima"])
            print(f"SARIMA -> RMSE: {rmse_sarima:.2f}, MAPE: {mape_sarima:.2f}%")
        except Exception as e:
            print("Error evaluating SARIMA:", e)
    else:
        print("SARIMA skipped (statsmodels missing or failed).")

    # XGBoost or RandomForest fallback
    features = [c for c in data.columns if c not in ["sales"]]
    pred_series, model_name = try_xgboost_or_rf(train, test, features)
    if pred_series is not None:
        col_name = model_name
        results[col_name] = pred_series.reindex(results.index)
        try:
            rmse_model, mape_model = evaluate(results["sales"], results[col_name])
            print(f"{col_name} -> RMSE: {rmse_model:.2f}, MAPE: {mape_model:.2f}%")
        except Exception as e:
            print(f"Error evaluating {col_name} predictions:", e)
    else:
        print("No tree-based model predictions produced.")

    # Save
    results.to_csv(OUT_FILE)
    print(f"Forecast results written to {OUT_FILE} in current folder.")


if __name__ == "__main__":
    main()
