import pandas as pd
import numpy as np
from pathlib import Path
import sys
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import mlflow.statsmodels
import os
import warnings

# Allow running this file directly: `python src/models/arima_model.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_conf import configure_mlflow

# Set up MLflow
mlflow.set_tracking_uri(configure_mlflow())
mlflow.set_experiment("BTCUSD_ARIMA_Forecasting")

# Suppress ARIMA warnings
warnings.filterwarnings("ignore")


def _safe_mape_pct(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    non_zero = np.abs(y_true_arr) > 1e-8
    if not np.any(non_zero):
        return None
    return float(np.mean(np.abs((y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero])) * 100.0)


def _select_arima_order(series_train, p_values=(0, 1, 2, 3), d_values=(1,), q_values=(0, 1, 2, 3)):
    best_order = None
    best_aic = float("inf")

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    fit = ARIMA(
                        series_train,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()
                    if fit.aic < best_aic:
                        best_aic = float(fit.aic)
                        best_order = (p, d, q)
                except Exception:
                    continue

    return best_order if best_order is not None else (1, 1, 1), best_aic

def train_arima_model(p=1, d=1, q=1):
    """
    Trains an ARIMA model on BTCUSD Close price.
    p: Lag order
    d: Degree of differencing
    q: Order of moving average
    """
    # Load processed data
    data_path = "data/processed/btcusd_processed.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run ingestion.py first.")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # ARIMA uses univariate time series
    series = df['Close']
    
    # Split data (80% train, 20% test)
    split_point = int(len(series) * 0.8)
    train, test = series[0:split_point], series[split_point:]

    selected_order, selected_aic = _select_arima_order(train)
    p, d, q = selected_order
    
    with mlflow.start_run(run_name="ARIMA_Model"):
        # Log parameters
        mlflow.log_param("model_type", "ARIMA")
        mlflow.log_param("p", p)
        mlflow.log_param("d", d)
        mlflow.log_param("q", q)
        if np.isfinite(selected_aic):
            mlflow.log_metric("selected_aic", float(selected_aic))
        
        # Fit model
        print(f"Training ARIMA({p},{d},{q})...")
        # A) Evaluation fit on split for comparable MSE.
        eval_model = ARIMA(
            train,
            order=(p, d, q),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        eval_fit = eval_model.fit()
        predictions = eval_fit.forecast(steps=len(test))
        mse = mean_squared_error(test, predictions)
        rmse_usd = float(np.sqrt(mse))

        mape_pct = _safe_mape_pct(test.values, predictions.values)
        accuracy_pct = max(0.0, 100.0 - mape_pct) if mape_pct is not None else None

        # B) Production fit on full series for latest endpoint next-day forecast.
        full_model = ARIMA(
            series,
            order=(p, d, q),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        full_fit = full_model.fit()
        next_day_pred = float(full_fit.forecast(steps=1).iloc[0])

        # Log values for dashboard and diagnostics.
        mlflow.log_metric("mse", float(mse))
        mlflow.log_metric("rmse_usd", rmse_usd)
        if mape_pct is not None:
            mlflow.log_metric("mape_pct", mape_pct)
            mlflow.log_metric("accuracy_pct", accuracy_pct)
        mlflow.log_metric("arima_next_day_pred", next_day_pred)
        mlflow.log_metric("arima_last_close", float(series.iloc[-1]))

        mlflow.statsmodels.log_model(
            full_fit,
            artifact_path="model",
            registered_model_name="BTCUSD_ARIMA_Model",
        )
        
        print(f"ARIMA Training complete.")
        print(f"Selected ARIMA order: ({p},{d},{q})")
        if mape_pct is not None:
            print(
                f"MSE: {mse:.4f}, RMSE: {rmse_usd:.2f}, MAPE: {mape_pct:.2f}%, "
                f"Accuracy: {accuracy_pct:.2f}%, Next-Day Pred: {next_day_pred:.2f}"
            )
        else:
            print(f"MSE: {mse:.4f}, RMSE: {rmse_usd:.2f}, Next-Day Pred: {next_day_pred:.2f}")
        
        return full_fit, mse

if __name__ == "__main__":
    try:
        train_arima_model()
    except Exception as e:
        print(f"Training failed: {e}")
