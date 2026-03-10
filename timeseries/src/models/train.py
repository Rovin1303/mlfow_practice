import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import mlflow
import os

# Allow running this file directly: `python src/models/train.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_conf import configure_mlflow

# Explicitly import mlflow.tensorflow for autologging
import mlflow.tensorflow

# Set up MLflow
# Using local tracking for now to avoid server connectivity issues in sandbox
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri(configure_mlflow())
mlflow.set_experiment("BTCUSD_Forecasting")
mlflow.tensorflow.autolog()


def _inverse_close_from_scaled(scaled_close, scaler):
    close_min = float(scaler.data_min_[0])
    close_max = float(scaler.data_max_[0])
    if close_max == close_min:
        return close_min
    return close_min + (float(scaled_close) * (close_max - close_min))


def _safe_mape_percent(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    non_zero = np.abs(y_true_arr) > 1e-8
    if not np.any(non_zero):
        return None
    mape = np.mean(np.abs((y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero])) * 100.0
    return float(mape)


def _stabilize_next_day_prediction(raw_pred, latest_close, residual_bias, recent_returns):
    # Correct systematic under/over prediction bias from recent validation behavior.
    corrected = float(raw_pred) + float(residual_bias)

    if not isinstance(latest_close, (int, float)) or latest_close <= 0:
        return corrected

    returns_arr = np.asarray(recent_returns, dtype=float)
    returns_arr = returns_arr[np.isfinite(returns_arr)]
    if returns_arr.size == 0:
        max_move = 0.08
    else:
        # Keep one-day movement in a realistic regime: about 2.5 sigma, bounded [3%, 12%].
        sigma = float(np.std(returns_arr))
        max_move = float(np.clip(2.5 * sigma, 0.03, 0.12))

    lower = float(latest_close) * (1.0 - max_move)
    upper = float(latest_close) * (1.0 + max_move)
    return float(np.clip(corrected, lower, upper))

def prepare_sequences(data, target_col='Close', window_size=60):
    """
    Prepare sequences for RNN training.
    """
    scaler = MinMaxScaler()
    # Use features: Close, MA7, MA21, Daily_Return, Volume
    features = ['Close', 'MA7', 'MA21', 'Daily_Return', 'Volume']
    scaled_data = scaler.fit_transform(data[features])
    
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i, 0])  # Predicting 'Close' price
        
    return np.array(X), np.array(y), scaler

def build_model(input_shape, model_type='LSTM', units=50, dropout=0.2):
    """
    Build LSTM or GRU model using modern Keras Input object.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    if model_type == 'LSTM':
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units, return_sequences=False))
        model.add(Dropout(dropout))
    else:
        model.add(GRU(units=units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(GRU(units=units, return_sequences=False))
        model.add(Dropout(dropout))
        
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model_type='LSTM', window_size=60, epochs=1, batch_size=64):
    # Load processed data
    data_path = "data/processed/btcusd_processed.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run ingestion.py first.")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    X, y, scaler = prepare_sequences(df, window_size=window_size)
    
    # Time-aware split: train / validation / test = 70 / 15 / 15.
    n_total = len(X)
    train_end = int(0.70 * n_total)
    val_end = int(0.85 * n_total)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Insufficient sequence rows for train/val/test split. Add more data or lower window_size.")
    
    # MLflow tracking
    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("split", "70_15_15")
        
        model = build_model((X_train.shape[1], X_train.shape[2]), model_type=model_type)
        
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        ]

        # Train
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss = model.evaluate(X_test, y_test)
        mlflow.log_metric("test_mse", test_loss)

        # Convert split-test predictions back to USD so metrics are comparable across models.
        y_test_pred_scaled = model.predict(X_test, verbose=0).reshape(-1)
        y_test_usd = np.array([_inverse_close_from_scaled(val, scaler) for val in y_test])
        y_test_pred_usd = np.array([_inverse_close_from_scaled(val, scaler) for val in y_test_pred_scaled])
        residual_bias = float(np.mean(y_test_usd - y_test_pred_usd))
        rmse_usd = float(np.sqrt(np.mean((y_test_usd - y_test_pred_usd) ** 2)))
        mape_pct = _safe_mape_percent(y_test_usd, y_test_pred_usd)
        actual_direction = np.sign(np.diff(y_test_usd))
        predicted_direction = np.sign(np.diff(y_test_pred_usd))
        direction_acc_pct = float((actual_direction == predicted_direction).mean() * 100.0) if len(actual_direction) else 0.0

        mlflow.log_metric("rmse_usd", rmse_usd)
        mlflow.log_metric("direction_acc_pct", direction_acc_pct)
        if mape_pct is not None:
            mlflow.log_metric("mape_pct", mape_pct)
            mlflow.log_metric("accuracy_pct", max(0.0, 100.0 - mape_pct))

        # Production-style next-day forecast from the latest available window.
        features = ['Close', 'MA7', 'MA21', 'Daily_Return', 'Volume']
        scaled_full = scaler.transform(df[features])
        latest_window = scaled_full[-window_size:]
        x_latest = np.expand_dims(latest_window, axis=0)

        pred_scaled = model.predict(x_latest, verbose=0)
        pred_scaled_value = float(np.array(pred_scaled).reshape(-1)[0])
        raw_next_day_pred = _inverse_close_from_scaled(pred_scaled_value, scaler)
        latest_close = float(df["Close"].iloc[-1])
        recent_returns = df["Daily_Return"].tail(90).values if "Daily_Return" in df.columns else []
        rnn_next_day_pred = _stabilize_next_day_prediction(
            raw_next_day_pred,
            latest_close,
            residual_bias,
            recent_returns,
        )

        mlflow.log_metric("rnn_next_day_pred", float(rnn_next_day_pred))
        mlflow.log_metric("rnn_next_day_pred_raw", float(raw_next_day_pred))
        mlflow.log_metric("latest_close", latest_close)
        
        # Log model with registry
        model_info = mlflow.tensorflow.log_model(
            model, 
            "model",
            registered_model_name="BTCUSD_RNN_Model"
        )
        
        print(f"Training complete. Test Loss (MSE): {test_loss}")
        print(
            f"Comparable metrics -> RMSE (USD): {rmse_usd:.2f}, MAPE: {mape_pct:.2f}%, Direction Acc: {direction_acc_pct:.2f}%"
            if mape_pct is not None
            else f"Comparable metrics -> RMSE (USD): {rmse_usd:.2f}, Direction Acc: {direction_acc_pct:.2f}%"
        )
        print(f"Model registered as 'BTCUSD_RNN_Model' at: {model_info.model_uri}")
        return model, history

if __name__ == "__main__":
    # Ensure MLflow server is running or use local tracking
    # For this exercise, we'll use default local tracking if no server
    try:
        tf.random.set_seed(42)
        np.random.seed(42)
        train_model(model_type='LSTM', epochs=30)
    except Exception as e:
        print(f"Training failed: {e}")
        print("Note: Ensure MLflow tracking server is reachable if set.")
