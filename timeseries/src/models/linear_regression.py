import argparse
import os
from pathlib import Path
import sys
from datetime import timedelta

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Allow running this file directly: `python src/models/linear_regression.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_conf import configure_mlflow

mlflow.set_tracking_uri(configure_mlflow())
mlflow.set_experiment("BTCUSD_Linear_Regression")


def load_dataset():
    data_path = "data/processed/btcusd_processed.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run ingestion first.")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # Add lagged/derived price structure to make the linear model more expressive.
    for lag in [1, 2, 3, 5, 7]:
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
    df["MA_gap_7_21"] = df["MA7"] - df["MA21"]
    df["Volatility_7"] = df["Daily_Return"].rolling(7).std()
    df["Target"] = df["Close"].shift(-1)
    df = df.dropna()
    return df


def _safe_mape_pct(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    non_zero = np.abs(y_true_arr) > 1e-8
    if not np.any(non_zero):
        return None
    return float(np.mean(np.abs((y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero])) * 100.0)


def _make_ridge_model(alpha):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _select_best_alpha(X_train, y_train):
    # Use the most recent slice of training as validation to avoid temporal leakage.
    val_size = max(30, int(0.2 * len(X_train)))
    X_fit, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_fit, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]
    if len(X_fit) < 30:
        return 1.0

    candidate_alphas = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0]
    best_alpha = 1.0
    best_mape = float("inf")

    for alpha in candidate_alphas:
        model = _make_ridge_model(alpha)
        model.fit(X_fit, y_fit)
        pred_val = model.predict(X_val)
        mape_val = _safe_mape_pct(y_val.values, pred_val)
        if mape_val is None:
            continue
        if mape_val < best_mape:
            best_mape = mape_val
            best_alpha = alpha

    return float(best_alpha)


def train_linear_regression(test_size=0.2):
    df = load_dataset()
    features = [
        "Close",
        "MA7",
        "MA21",
        "Daily_Return",
        "Volume",
        "Close_lag_1",
        "Close_lag_2",
        "Close_lag_3",
        "Close_lag_5",
        "Close_lag_7",
        "MA_gap_7_21",
        "Volatility_7",
    ]

    split = int((1 - test_size) * len(df))
    X_train, X_test = df[features].iloc[:split], df[features].iloc[split:]
    y_train, y_test = df["Target"].iloc[:split], df["Target"].iloc[split:]

    with mlflow.start_run(run_name="Linear_Regression_Baseline"):
        best_alpha = _select_best_alpha(X_train, y_train)

        mlflow.log_param("model_type", "RidgeRegression")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("features", features)
        mlflow.log_param("alpha", best_alpha)

        # Evaluation model on split data for comparable validation metrics.
        eval_model = _make_ridge_model(best_alpha)
        eval_model.fit(X_train, y_train)
        pred = eval_model.predict(X_test)

        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mse))
        non_zero = np.abs(y_test.values) > 1e-8
        if np.any(non_zero):
            mape_pct = float(np.mean(np.abs((y_test.values[non_zero] - pred[non_zero]) / y_test.values[non_zero])) * 100.0)
            accuracy_pct = max(0.0, 100.0 - mape_pct)
        else:
            mape_pct = None
            mean_close = float(X_test["Close"].mean())
            accuracy_pct = max(0.0, 100.0 * (1.0 - rmse / mean_close))

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("rmse_usd", rmse)
        if mape_pct is not None:
            mlflow.log_metric("mape_pct", mape_pct)
        mlflow.log_metric("accuracy_pct", accuracy_pct)

        # Final model for production-style next day forecast from latest features.
        final_model = _make_ridge_model(best_alpha)
        final_model.fit(df[features], df["Target"])
        latest_features = df[features].iloc[[-1]]
        linear_next_day_pred = float(final_model.predict(latest_features)[0])

        mlflow.log_metric("linear_next_day_pred", linear_next_day_pred)
        mlflow.log_metric("latest_close", float(df["Close"].iloc[-1]))

        mlflow.sklearn.log_model(
            final_model, "model", registered_model_name="BTCUSD_Linear_Regression"
        )

        print(
            f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
            f"MAPE: {mape_pct:.2f}% , Accuracy: {accuracy_pct:.2f}%, Alpha: {best_alpha}, Next-Day Pred: {linear_next_day_pred:.2f}"
            if mape_pct is not None
            else f"Accuracy: {accuracy_pct:.2f}%, Alpha: {best_alpha}, Next-Day Pred: {linear_next_day_pred:.2f}"
        )
        return final_model


def backtest_linear_regression(years=2, initial_train_days=180, step_days=30, horizon_days=30):
    df = load_dataset()
    features = ["Close", "MA7", "MA21", "Daily_Return", "Volume"]

    end_date = df.index.max()
    start_date = end_date - pd.DateOffset(years=years)
    bt = df.loc[df.index >= start_date].copy()

    if len(bt) < (initial_train_days + horizon_days + 30):
        raise ValueError("Not enough rows for the selected backtest window.")

    rows = []
    fold = 0
    split_date = bt.index.min() + timedelta(days=initial_train_days)

    while split_date + timedelta(days=horizon_days) <= bt.index.max():
        train = bt.loc[bt.index < split_date]
        test = bt.loc[(bt.index >= split_date) & (bt.index < split_date + timedelta(days=horizon_days))]

        if len(train) < 30 or len(test) < 5:
            split_date += timedelta(days=step_days)
            continue

        model = _make_ridge_model(1.0)
        model.fit(train[features], train["Target"])
        pred = model.predict(test[features])

        mse = mean_squared_error(test["Target"], pred)
        mae = mean_absolute_error(test["Target"], pred)
        rmse = float(np.sqrt(mse))
        mean_close = float(test["Close"].mean())
        accuracy_pct = max(0.0, 100.0 * (1.0 - rmse / mean_close))

        # Directional accuracy: did we predict up/down correctly vs current close
        actual_dir = np.sign(test["Target"].values - test["Close"].values)
        pred_dir = np.sign(pred - test["Close"].values)
        direction_acc = float((actual_dir == pred_dir).mean() * 100.0)

        rows.append(
            {
                "fold": fold,
                "train_end": split_date.strftime("%Y-%m-%d"),
                "test_points": len(test),
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "accuracy_pct": accuracy_pct,
                "direction_acc_pct": direction_acc,
            }
        )

        fold += 1
        split_date += timedelta(days=step_days)

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("No valid folds generated.")

    summary = {
        "folds": int(len(results)),
        "avg_mse": float(results["mse"].mean()),
        "avg_mae": float(results["mae"].mean()),
        "avg_rmse": float(results["rmse"].mean()),
        "avg_accuracy_pct": float(results["accuracy_pct"].mean()),
        "avg_direction_acc_pct": float(results["direction_acc_pct"].mean()),
    }

    print("\nBacktest Summary (2 years):")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    results.to_csv("data/processed/linear_regression_backtest_2y.csv", index=False)
    print("Saved fold metrics: data/processed/linear_regression_backtest_2y.csv")

    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--initial-train-days", type=int, default=180)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--horizon-days", type=int, default=30)
    args = parser.parse_args()

    if args.backtest:
        backtest_linear_regression(
            years=args.years,
            initial_train_days=args.initial_train_days,
            step_days=args.step_days,
            horizon_days=args.horizon_days,
        )
    else:
        train_linear_regression()