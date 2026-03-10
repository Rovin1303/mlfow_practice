from django.shortcuts import render
import math
from pathlib import Path
import sys

from mlflow.tracking import MlflowClient
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_conf import DEFAULT_EXPERIMENTS, MLFLOW_TRACKING_URI, PROCESSED_BTCUSD_CSV


MODEL_DEFS = [
    {
        "key": "rnn",
        "label": "RNN (LSTM)",
        "experiment": "BTCUSD_Forecasting",
        "registry_model": "BTCUSD_RNN_Model",
        "mse_keys": ["test_mse", "mse"],
        "rmse_keys": ["rmse_usd", "rmse"],
        "mape_keys": ["mape_pct"],
        "accuracy_keys": ["accuracy_pct"],
        "prediction_keys": ["rnn_next_day_pred", "next_day_pred"],
    },
    {
        "key": "linear",
        "label": "Linear Regression",
        "experiment": "BTCUSD_Linear_Regression",
        "registry_model": "BTCUSD_Linear_Regression",
        "mse_keys": ["mse", "test_mse"],
        "rmse_keys": ["rmse_usd", "rmse"],
        "mape_keys": ["mape_pct"],
        "accuracy_keys": ["accuracy_pct"],
        "prediction_keys": ["linear_next_day_pred", "next_day_pred"],
    },
    {
        "key": "arima",
        "label": "ARIMA",
        "experiment": "BTCUSD_ARIMA_Forecasting",
        "registry_model": "BTCUSD_ARIMA_Model",
        "mse_keys": ["mse", "test_mse"],
        "rmse_keys": ["rmse_usd", "rmse"],
        "mape_keys": ["mape_pct"],
        "accuracy_keys": ["accuracy_pct"],
        "prediction_keys": ["arima_next_day_pred", "next_day_pred"],
    },
]


def _read_processed_data():
    try:
        if PROCESSED_BTCUSD_CSV.exists():
            return pd.read_csv(PROCESSED_BTCUSD_CSV)
    except Exception:
        pass
    return None


def _extract_close_reference(df):
    if df is None or df.empty:
        return None
    for column in df.columns:
        if column.lower() == "close":
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if not series.empty:
                return float(series.mean())
    return None


def _extract_latest_btc_price(df):
    if df is None or df.empty:
        return None
    for column in df.columns:
        if column.lower() == "close":
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[-1])
    return None


def _compute_accuracy_percent(accuracy_value, mape_value, rmse_value, mse_value, close_reference):
    if isinstance(accuracy_value, (int, float)):
        return max(0.0, min(100.0, float(accuracy_value)))
    if isinstance(mape_value, (int, float)):
        return max(0.0, min(100.0, 100.0 - float(mape_value)))

    fallback_rmse = None
    if isinstance(rmse_value, (int, float)) and rmse_value >= 0:
        fallback_rmse = float(rmse_value)
    elif isinstance(mse_value, (int, float)) and mse_value >= 0:
        fallback_rmse = math.sqrt(float(mse_value))

    if fallback_rmse is not None and close_reference and close_reference > 0:
        accuracy = 100.0 * (1.0 - (fallback_rmse / close_reference))
        return max(0.0, min(100.0, accuracy))
    return None


def _pick_first_numeric(metrics, keys):
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _weighted_average(values):
    weighted_sum = 0.0
    total_weight = 0.0
    for value, weight in values:
        if not isinstance(value, (int, float)):
            continue
        if not isinstance(weight, (int, float)) or weight <= 0:
            continue
        weighted_sum += float(value) * float(weight)
        total_weight += float(weight)

    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def _compute_model_quality(mape_value, rmse_value, mse_value, prediction_value, latest_btc_price, close_reference):
    if not isinstance(prediction_value, (int, float)):
        return None

    if close_reference and close_reference > 0 and isinstance(rmse_value, (int, float)) and rmse_value >= 0:
        rmse_component = float(rmse_value) / close_reference
    elif close_reference and close_reference > 0 and isinstance(mse_value, (int, float)) and mse_value >= 0:
        rmse_component = math.sqrt(float(mse_value)) / close_reference
    else:
        rmse_component = 1.0

    if latest_btc_price and latest_btc_price > 0:
        deviation_ratio = abs(float(prediction_value) - float(latest_btc_price)) / float(latest_btc_price)
    else:
        deviation_ratio = 0.0

    # Lower score is better; choose model whose next-day forecast is nearest to market.
    if latest_btc_price and latest_btc_price > 0:
        return deviation_ratio

    # Fallback only when no latest market anchor is available.
    return rmse_component


def _collect_registry_statuses(client: MlflowClient):
    statuses = []
    try:
        registered_models = list(client.search_registered_models())
    except Exception:
        return statuses

    for registered_model in registered_models:
        latest_versions = list(getattr(registered_model, "latest_versions", []) or [])
        latest_versions.sort(key=lambda item: int(getattr(item, "version", 0)), reverse=True)

        latest_version = "-"
        latest_stage = "None"
        if latest_versions:
            latest_version = str(getattr(latest_versions[0], "version", "-"))
            latest_stage = getattr(latest_versions[0], "current_stage", "None") or "None"

        statuses.append(
            {
                "name": getattr(registered_model, "name", "-"),
                "versions": len(latest_versions),
                "latest_version": latest_version,
                "latest_stage": latest_stage,
            }
        )

    statuses.sort(key=lambda item: item["name"].lower())
    return statuses


def _get_finished_runs(client: MlflowClient, experiment_name, max_results=30):
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=max_results,
    )
    return list(runs or [])


def _infer_prediction_from_registry(client: MlflowClient, model_name, processed_df):
    if processed_df is None or processed_df.empty:
        return None

    try:
        import mlflow.pyfunc
    except Exception:
        return None

    try:
        registered_model = client.get_registered_model(model_name)
        sorted_versions = list(getattr(registered_model, "latest_versions", []) or [])
        sorted_versions.sort(key=lambda item: int(getattr(item, "version", 0)), reverse=True)
    except Exception:
        return None

    for version in sorted_versions:
        model_version = getattr(version, "version", None)
        if not model_version:
            continue

        model_uri = f"models:/{model_name}/{model_version}"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            last_row = processed_df.tail(1).copy()
            prediction = model.predict(last_row)
            if len(prediction) > 0:
                return float(prediction[0])
        except Exception:
            continue
    return None


def _infer_linear_from_registry(client: MlflowClient, model_name, processed_df):
    if processed_df is None or processed_df.empty:
        return None

    feature_cols = ["Close", "MA7", "MA21", "Daily_Return", "Volume"]
    if not all(col in processed_df.columns for col in feature_cols):
        return None

    try:
        import mlflow.pyfunc

        registered_model = client.get_registered_model(model_name)
        versions = list(getattr(registered_model, "latest_versions", []) or [])
        versions.sort(key=lambda item: int(getattr(item, "version", 0)), reverse=True)
    except Exception:
        return None

    latest_row = processed_df[feature_cols].tail(1).copy()
    for version in versions:
        model_version = getattr(version, "version", None)
        if not model_version:
            continue
        try:
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
            pred = model.predict(latest_row)
            if len(pred) > 0:
                return float(pred[0])
        except Exception:
            continue
    return None


def _infer_rnn_from_registry(client: MlflowClient, model_name, processed_df, window_size=60):
    if processed_df is None or processed_df.empty:
        return None

    feature_cols = ["Close", "MA7", "MA21", "Daily_Return", "Volume"]
    if not all(col in processed_df.columns for col in feature_cols):
        return None
    if len(processed_df) < window_size:
        return None

    try:
        import mlflow.tensorflow
        from sklearn.preprocessing import MinMaxScaler

        registered_model = client.get_registered_model(model_name)
        versions = list(getattr(registered_model, "latest_versions", []) or [])
        versions.sort(key=lambda item: int(getattr(item, "version", 0)), reverse=True)
    except Exception:
        return None

    frame = processed_df[feature_cols].copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(frame)
    x = scaled[-window_size:]
    x = x.reshape(1, window_size, len(feature_cols))

    for version in versions:
        model_version = getattr(version, "version", None)
        if not model_version:
            continue
        try:
            model = mlflow.tensorflow.load_model(f"models:/{model_name}/{model_version}")
            pred_scaled = model.predict(x, verbose=0)
            pred_scaled_value = float(pred_scaled.reshape(-1)[0])

            close_min = float(scaler.data_min_[0])
            close_max = float(scaler.data_max_[0])
            if close_max == close_min:
                return close_min
            return close_min + pred_scaled_value * (close_max - close_min)
        except Exception:
            continue
    return None


def _infer_prediction_from_registry_for_model(client: MlflowClient, model_def, processed_df):
    model_name = model_def.get("registry_model")
    if not model_name:
        return None

    if model_def["key"] == "linear":
        return _infer_linear_from_registry(client, model_name, processed_df)
    if model_def["key"] == "rnn":
        return _infer_rnn_from_registry(client, model_name, processed_df)

    return _infer_prediction_from_registry(client, model_name, processed_df)


def _get_model_snapshot(client: MlflowClient, model_def, processed_df, close_reference, latest_btc_price):
    snapshot = {
        "key": model_def["key"],
        "label": model_def["label"],
        "status": "N/A",
        "run_id": None,
        "start_time": None,
        "prediction": None,
        "mse": None,
        "rmse_usd": None,
        "mape_pct": None,
        "market_gap_pct": None,
        "accuracy_pct": None,
        "meta": "",
        "prediction_source": "unavailable",
        "quality_score": None,
        "is_best": False,
    }

    finished_runs = _get_finished_runs(client, model_def["experiment"], max_results=30)
    if not finished_runs:
        return snapshot

    latest_run = finished_runs[0]
    latest_params = latest_run.data.params or {}

    snapshot["status"] = "FINISHED"
    snapshot["run_id"] = latest_run.info.run_id[:8]
    snapshot["start_time"] = pd.to_datetime(latest_run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M")

    latest_metrics = latest_run.data.metrics or {}

    mse_candidates = []
    rmse_candidates = []
    mape_candidates = []
    for idx, run in enumerate(finished_runs[:5]):
        metrics = run.data.metrics or {}
        mse_value = _pick_first_numeric(metrics, model_def["mse_keys"])
        rmse_value = _pick_first_numeric(metrics, model_def["rmse_keys"])
        mape_value = _pick_first_numeric(metrics, model_def["mape_keys"])
        recency_weight = 0.85 ** idx

        if isinstance(mse_value, (int, float)) and mse_value > 0:
            mse_candidates.append((float(mse_value), recency_weight))
        if isinstance(rmse_value, (int, float)) and rmse_value >= 0:
            rmse_candidates.append((float(rmse_value), recency_weight))
        if isinstance(mape_value, (int, float)) and mape_value >= 0:
            mape_candidates.append((float(mape_value), recency_weight))

    mse_value = _weighted_average(mse_candidates)
    rmse_value = _weighted_average(rmse_candidates)
    mape_value = _weighted_average(mape_candidates)

    prediction_value = _pick_first_numeric(latest_metrics, model_def["prediction_keys"])
    if isinstance(prediction_value, (int, float)):
        snapshot["prediction_source"] = "metric"

    if prediction_value is None:
        prediction_value = _infer_prediction_from_registry_for_model(client, model_def, processed_df)
        if isinstance(prediction_value, (int, float)):
            snapshot["prediction_source"] = "registry-infer"

    if prediction_value is None and model_def["key"] == "arima":
        fallback_arima_pred = latest_metrics.get("arima_next_day_pred")
        if isinstance(fallback_arima_pred, (int, float)):
            prediction_value = float(fallback_arima_pred)
            snapshot["prediction_source"] = "metric"

    explicit_accuracy = _pick_first_numeric(latest_metrics, model_def["accuracy_keys"])
    accuracy_value = _compute_accuracy_percent(
        explicit_accuracy,
        mape_value,
        rmse_value,
        mse_value,
        close_reference,
    )

    snapshot["mse"] = mse_value
    snapshot["rmse_usd"] = rmse_value
    snapshot["mape_pct"] = mape_value
    snapshot["prediction"] = float(prediction_value) if isinstance(prediction_value, (int, float)) else None
    if isinstance(snapshot["prediction"], (int, float)) and isinstance(latest_btc_price, (int, float)) and latest_btc_price > 0:
        snapshot["market_gap_pct"] = (
            abs(float(snapshot["prediction"]) - float(latest_btc_price))
            / float(latest_btc_price)
            * 100.0
        )
    snapshot["accuracy_pct"] = accuracy_value
    snapshot["meta"] = (
        f"order={latest_params.get('p', '?')},{latest_params.get('d', '?')},{latest_params.get('q', '?')}"
        if model_def["key"] == "arima"
        else ""
    )
    snapshot["quality_score"] = _compute_model_quality(
        mape_value,
        rmse_value,
        mse_value,
        snapshot["prediction"],
        latest_btc_price,
        close_reference,
    )

    return snapshot


def _get_recent_runs(client: MlflowClient):
    rows = []
    for experiment_name in DEFAULT_EXPERIMENTS:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            continue

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=10,
        )

        for run in runs:
            metrics = run.data.metrics or {}
            params = run.data.params or {}
            mse_value = _pick_first_numeric(metrics, ["mse", "test_mse"])
            ts = pd.to_datetime(run.info.start_time, unit="ms")

            rows.append(
                {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "model_type": params.get("model_type", experiment_name),
                    "mse": mse_value,
                    "start_time": ts.strftime("%Y-%m-%d %H:%M"),
                    "_start_time": ts,
                }
            )

    rows.sort(key=lambda row: row["_start_time"], reverse=True)
    trimmed = rows[:8]
    for row in trimmed:
        row.pop("_start_time", None)
    return trimmed


def _select_best_model(model_snapshots):
    candidates = [
        snapshot
        for snapshot in model_snapshots
        if isinstance(snapshot.get("prediction"), (int, float)) and isinstance(snapshot.get("quality_score"), (int, float))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["quality_score"])


def _build_drift_summary(processed_df, reference_window=90, current_window=30):
    summary = {
        "status": "Insufficient Data",
        "score": None,
        "mean_shift_pct": None,
        "return_shift_pct": None,
        "volatility_ratio": None,
        "reference_window": reference_window,
        "current_window": current_window,
        "sample_count": 0,
    }

    if processed_df is None or processed_df.empty or "Close" not in processed_df.columns:
        return summary

    close_series = pd.to_numeric(processed_df["Close"], errors="coerce").dropna()
    summary["sample_count"] = int(len(close_series))

    if len(close_series) < (reference_window + current_window):
        return summary

    lookback = close_series.iloc[-(reference_window + current_window) :]
    reference = lookback.iloc[:reference_window]
    current = lookback.iloc[reference_window:]

    ref_mean = float(reference.mean())
    cur_mean = float(current.mean())
    mean_shift_pct = ((cur_mean - ref_mean) / ref_mean * 100.0) if ref_mean > 0 else 0.0

    ref_returns = reference.pct_change().dropna()
    cur_returns = current.pct_change().dropna()

    ref_return_mean = float(ref_returns.mean()) if not ref_returns.empty else 0.0
    cur_return_mean = float(cur_returns.mean()) if not cur_returns.empty else 0.0
    return_shift_pct = (cur_return_mean - ref_return_mean) * 100.0

    ref_vol = float(ref_returns.std()) if not ref_returns.empty else 0.0
    cur_vol = float(cur_returns.std()) if not cur_returns.empty else 0.0
    volatility_ratio = (cur_vol / ref_vol) if ref_vol > 0 else None

    score = abs(mean_shift_pct) * 0.45 + abs(return_shift_pct) * 2.5
    if isinstance(volatility_ratio, (int, float)):
        score += abs((volatility_ratio - 1.0) * 100.0) * 0.20

    if score < 4.0:
        status = "Low Drift"
    elif score < 9.0:
        status = "Moderate Drift"
    else:
        status = "High Drift"

    summary.update(
        {
            "status": status,
            "score": float(score),
            "mean_shift_pct": float(mean_shift_pct),
            "return_shift_pct": float(return_shift_pct),
            "volatility_ratio": float(volatility_ratio) if isinstance(volatility_ratio, (int, float)) else None,
        }
    )
    return summary


def dashboard_overview(request):
    """
    Fetches experiment data and model registry info from MLflow for the dashboard.
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    processed_df = _read_processed_data()
    close_reference = _extract_close_reference(processed_df)
    latest_btc_price = _extract_latest_btc_price(processed_df)

    registry_statuses = _collect_registry_statuses(client)
    model_snapshots = [
        _get_model_snapshot(
            client,
            model_def,
            processed_df,
            close_reference,
            latest_btc_price,
        )
        for model_def in MODEL_DEFS
    ]

    best_model_snapshot = _select_best_model(model_snapshots)
    if best_model_snapshot:
        best_model_snapshot["is_best"] = True

    model_accuracy = None
    if best_model_snapshot:
        model_accuracy = best_model_snapshot.get("accuracy_pct")

    runs_data = _get_recent_runs(client)

    context = {
        "model_accuracy": f"{model_accuracy:.2f}%" if isinstance(model_accuracy, (int, float)) else "N/A",
        "latest_btc_price": f"{latest_btc_price:.2f}" if isinstance(latest_btc_price, (int, float)) else "N/A",
        "runs": runs_data,
        "model_snapshots": model_snapshots,
        "best_model": best_model_snapshot,
        "registry_statuses": registry_statuses,
        "registered_models_count": len(registry_statuses),
        "registered_versions_count": sum(item["versions"] for item in registry_statuses),
    }

    return render(request, "dashboard/overview.html", context)


def drift_monitoring(request):
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    processed_df = _read_processed_data()
    latest_btc_price = _extract_latest_btc_price(processed_df)
    close_reference = _extract_close_reference(processed_df)

    model_snapshots = [
        _get_model_snapshot(
            client,
            model_def,
            processed_df,
            close_reference,
            latest_btc_price,
        )
        for model_def in MODEL_DEFS
    ]
    best_model_snapshot = _select_best_model(model_snapshots)
    if best_model_snapshot:
        best_model_snapshot["is_best"] = True

    drift_summary = _build_drift_summary(processed_df)
    context = {
        "latest_btc_price": latest_btc_price,
        "drift": drift_summary,
        "model_snapshots": model_snapshots,
        "best_model": best_model_snapshot,
        "recent_runs": _get_recent_runs(client),
    }
    return render(request, "dashboard/drift_monitoring.html", context)
