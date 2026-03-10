import math
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import pandas as pd
import requests
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

from project_conf import MLFLOW_TRACKING_URI, PROCESSED_BTCUSD_CSV, PROJECT_ROOT


MODEL_WEIGHTS = {
    "lstm": 0.70,
    "arima": 0.20,
    "linear": 0.10,
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _fetch_fear_greed_index() -> Optional[float]:
    # Optional external sentiment signal (0..100). Fail silently.
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if data:
            return _safe_float(data[0].get("value"))
    except Exception:
        return None
    return None


def _compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out["Close"], errors="coerce")

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["BB_MID"] = ma20
    out["BB_UPPER"] = ma20 + 2 * std20
    out["BB_LOWER"] = ma20 - 2 * std20

    # Trend helpers
    out["MA50"] = close.rolling(50).mean()
    out["MA200"] = close.rolling(200).mean()

    # Volatility index proxy (annualized, %)
    returns = close.pct_change().dropna()
    vol = returns.rolling(30).std() * math.sqrt(365) * 100
    out["VOLATILITY_INDEX"] = vol.reindex(out.index)

    return out


def _load_latest_market_frame() -> pd.DataFrame:
    if not PROCESSED_BTCUSD_CSV.exists():
        raise FileNotFoundError(f"Processed data not found: {PROCESSED_BTCUSD_CSV}")

    df = pd.read_csv(PROCESSED_BTCUSD_CSV, index_col=0, parse_dates=True)
    required_cols = ["Close", "High", "Low", "Volume"]
    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = _compute_technical_indicators(df)
    df = df.sort_index()
    # Keep sufficient history for MA200 while serving latest 60-day logic.
    return df.tail(260)


def _get_experiment_mse(client: MlflowClient, experiment_name: str) -> Optional[float]:
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=20,
    )

    for run in runs:
        if run.info.status != "FINISHED":
            continue
        metrics = run.data.metrics or {}
        for key in ("mse", "test_mse"):
            value = _safe_float(metrics.get(key))
            if value is not None:
                return value
    return None


def _resolve_latest_model_uri(client: MlflowClient, model_name: str) -> Optional[str]:
    try:
        registered_model = client.get_registered_model(model_name)
        latest_versions = list(getattr(registered_model, "latest_versions", []) or [])
        latest_versions.sort(key=lambda item: int(getattr(item, "version", 0)), reverse=True)
        if latest_versions:
            return f"models:/{model_name}/{latest_versions[0].version}"
    except Exception:
        return None
    return None


def _resolve_latest_version(client: MlflowClient, model_name: str):
    try:
        registered_model = client.get_registered_model(model_name)
        latest_versions = list(getattr(registered_model, "latest_versions", []) or [])
        latest_versions.sort(key=lambda item: int(getattr(item, "version", 0)), reverse=True)
        if latest_versions:
            return latest_versions[0]
    except Exception:
        return None
    return None


def _find_local_model_dir_by_id(model_id: str) -> Optional[str]:
    root = Path(PROJECT_ROOT) / "mlruns"
    if not root.exists():
        return None
    for candidate in root.glob(f"**/models/{model_id}/artifacts"):
        mlmodel = candidate / "MLmodel"
        if mlmodel.exists():
            return str(candidate)
    return None


def _latest_local_model_dir(loader_module: str) -> Optional[str]:
    root = Path(PROJECT_ROOT) / "mlruns"
    if not root.exists():
        return None

    matches = []
    for mlmodel_file in root.glob("**/models/*/artifacts/MLmodel"):
        try:
            text = mlmodel_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if f"loader_module: {loader_module}" in text:
            matches.append(mlmodel_file.parent)

    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(matches[0])


def _predict_linear_regression(client: MlflowClient, latest_row: pd.DataFrame) -> Dict[str, Any]:
    result = {
        "prediction": None,
        "confidence": None,
        "caution": True,
        "mse": None,
        "valid": False,
        "error": None,
    }

    mse = _get_experiment_mse(client, "BTCUSD_Linear_Regression")
    result["mse"] = mse

    model = None
    try:
        model_uri = _resolve_latest_model_uri(client, "BTCUSD_Linear_Regression")
        if model_uri:
            model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        result["error"] = f"registry-load-failed:{exc.__class__.__name__}"

    if model is None:
        try:
            latest_version = _resolve_latest_version(client, "BTCUSD_Linear_Regression")
            source = getattr(latest_version, "source", "") if latest_version else ""
            model_id = source.split("/")[-1] if source.startswith("models:/") else None
            local_dir = _find_local_model_dir_by_id(model_id) if model_id else None
            if not local_dir:
                local_dir = _latest_local_model_dir("mlflow.sklearn")
            if local_dir:
                model = mlflow.sklearn.load_model(local_dir)
        except Exception as exc:
            if not result.get("error"):
                result["error"] = f"local-load-failed:{exc.__class__.__name__}"

    if model is None:
        return result

    try:
        features = ["Close", "MA7", "MA21", "Daily_Return", "Volume"]
        model_input = latest_row[features]
        prediction = _safe_float(model.predict(model_input)[0])
        result["prediction"] = prediction
        if prediction is not None and mse is not None and latest_row["Close"].iloc[-1] > 0:
            rmse = math.sqrt(mse)
            price = float(latest_row["Close"].iloc[-1])
            result["confidence"] = max(0.0, min(1.0, 1.0 - (rmse / price)))
            result["valid"] = True
    except Exception as exc:
        if not result.get("error"):
            result["error"] = f"predict-failed:{exc.__class__.__name__}"

    return result


def _predict_lstm(client: MlflowClient, frame: pd.DataFrame) -> Dict[str, Any]:
    result = {
        "prediction": None,
        "confidence": None,
        "mse": None,
        "valid": False,
        "error": None,
    }

    mse = _get_experiment_mse(client, "BTCUSD_Forecasting")
    result["mse"] = mse

    model = None
    try:
        model_uri = _resolve_latest_model_uri(client, "BTCUSD_RNN_Model")
        if model_uri:
            model = mlflow.tensorflow.load_model(model_uri)
    except Exception as exc:
        result["error"] = f"registry-load-failed:{exc.__class__.__name__}"

    if model is None:
        try:
            local_dir = _latest_local_model_dir("mlflow.tensorflow")
            if local_dir:
                model = mlflow.tensorflow.load_model(local_dir)
        except Exception as exc:
            if not result.get("error"):
                result["error"] = f"local-load-failed:{exc.__class__.__name__}"

    if model is None:
        return result

    try:
        features = ["Close", "MA7", "MA21", "Daily_Return", "Volume"]
        model_df = frame.dropna(subset=features).copy()
        if len(model_df) < 60:
            return result

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(model_df[features])
        x = np.array([scaled[-60:]])

        pred_scaled = model.predict(x, verbose=0)
        pred_scaled_value = float(np.asarray(pred_scaled).reshape(-1)[0])

        # Inverse using Close column scale boundaries from current window.
        close_min = float(model_df["Close"].min())
        close_max = float(model_df["Close"].max())
        prediction = close_min + (pred_scaled_value * (close_max - close_min))

        result["prediction"] = prediction

        if mse is not None and model_df["Close"].iloc[-1] > 0:
            rmse = math.sqrt(mse)
            price = float(model_df["Close"].iloc[-1])
            result["confidence"] = max(0.0, min(1.0, 1.0 - (rmse / price)))
        else:
            result["confidence"] = 0.65

        result["valid"] = True
    except Exception as exc:
        if not result.get("error"):
            result["error"] = f"predict-failed:{exc.__class__.__name__}"

    return result


def _predict_arima(client: MlflowClient, series: pd.Series) -> Dict[str, Any]:
    result = {
        "prediction": None,
        "confidence": None,
        "mse": None,
        "valid": False,
        "error": None,
    }

    mse = _get_experiment_mse(client, "BTCUSD_ARIMA_Forecasting")
    result["mse"] = mse

    # Requirement: only valid if MSE < 1000.
    if mse is None or mse >= 1000:
        result["error"] = "invalid-mse-threshold"
        return result

    try:
        model_fit = ARIMA(series, order=(1, 1, 1)).fit()
        prediction = _safe_float(model_fit.forecast(steps=1).iloc[0])
        result["prediction"] = prediction
        if prediction is not None and series.iloc[-1] > 0:
            rmse = math.sqrt(mse)
            result["confidence"] = max(0.0, min(1.0, 1.0 - (rmse / float(series.iloc[-1]))))
            result["valid"] = True
    except Exception as exc:
        result["error"] = f"predict-failed:{exc.__class__.__name__}"

    return result


def _apply_intelligent_adjustments(raw_prediction: float, df: pd.DataFrame) -> Dict[str, Any]:
    latest = df.iloc[-1]
    adjusted = raw_prediction
    adjustment_trace = []

    # 1) Volatility adjustment: ±5%
    vol_index = _safe_float(latest.get("VOLATILITY_INDEX"))
    if vol_index is not None:
        if vol_index >= 65:
            adjusted *= 0.95
            adjustment_trace.append("high-volatility:-5%")
        elif vol_index <= 35:
            adjusted *= 1.03
            adjustment_trace.append("low-volatility:+3%")

    # 2) Trend confirmation using MA50/MA200
    ma50 = _safe_float(latest.get("MA50"))
    ma200 = _safe_float(latest.get("MA200"))
    if ma50 is not None and ma200 is not None:
        if ma50 > ma200:
            adjusted *= 1.01
            adjustment_trace.append("bull-trend:+1%")
        else:
            adjusted *= 0.99
            adjustment_trace.append("bear-trend:-1%")

    # 3) Support/resistance proximity
    close_series = pd.to_numeric(df["Close"], errors="coerce").dropna()
    lookback = close_series.tail(30)
    if not lookback.empty:
        support = float(lookback.min())
        resistance = float(lookback.max())
        span = max(1.0, resistance - support)
        if abs(adjusted - resistance) / span < 0.10:
            adjusted *= 0.99
            adjustment_trace.append("near-resistance:-1%")
        elif abs(adjusted - support) / span < 0.10:
            adjusted *= 1.01
            adjustment_trace.append("near-support:+1%")

    # 4) Market sentiment (fear/greed)
    fear_greed = _fetch_fear_greed_index()
    if fear_greed is not None:
        if fear_greed <= 25:
            adjusted *= 0.99
            adjustment_trace.append("fear-greed:extreme-fear:-1%")
        elif fear_greed >= 75:
            adjusted *= 1.01
            adjustment_trace.append("fear-greed:extreme-greed:+1%")

    return {
        "adjusted_prediction": float(adjusted),
        "volatility_index": vol_index,
        "fear_greed_index": fear_greed,
        "adjustments": adjustment_trace,
    }


def _risk_from_volatility(volatility_index: Optional[float]) -> str:
    if volatility_index is None:
        return "medium"
    if volatility_index >= 65:
        return "high"
    if volatility_index <= 35:
        return "low"
    return "medium"


def _recommendation(prediction: float, latest_price: float, risk_level: str) -> str:
    if latest_price <= 0:
        return "hold"
    delta_pct = (prediction - latest_price) / latest_price
    if risk_level == "high":
        if delta_pct > 0.02:
            return "hold"
        if delta_pct < -0.02:
            return "sell"
        return "hold"

    if delta_pct > 0.01:
        return "buy"
    if delta_pct < -0.01:
        return "sell"
    return "hold"


def predict_next_day_btc_ensemble(log_to_mlflow: bool = True) -> Dict[str, Any]:
    """Generate next-day BTC prediction using LSTM/Linear/ARIMA ensemble with adjustments.

    Returns
    -------
    dict
        {
          next_day_prediction, confidence_interval_95, risk_assessment,
          recommendation, component_predictions, latest_btc_price,
          prediction_source
        }
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    frame = _load_latest_market_frame()
    latest_60 = frame.tail(60).copy()
    latest_row = latest_60.tail(1).copy()
    latest_price = float(pd.to_numeric(latest_row["Close"], errors="coerce").iloc[-1])

    lstm = _predict_lstm(client, frame)
    linear = _predict_linear_regression(client, latest_row)
    arima = _predict_arima(client, pd.to_numeric(frame["Close"], errors="coerce").dropna())

    # Weighting per requirement.
    lstm_w = MODEL_WEIGHTS["lstm"] if lstm.get("prediction") is not None else 0.0
    linear_w = MODEL_WEIGHTS["linear"] if linear.get("prediction") is not None else 0.0
    arima_w = (
        MODEL_WEIGHTS["arima"]
        if arima.get("valid") and arima.get("prediction") is not None
        else 0.0
    )

    weight_sum = lstm_w + linear_w + arima_w
    if weight_sum <= 0:
        raw_prediction = latest_price
        source = "fallback-latest-close"
    else:
        lstm_pred = float(lstm["prediction"]) if lstm.get("prediction") is not None else 0.0
        linear_pred = float(linear["prediction"]) if linear.get("prediction") is not None else 0.0
        arima_pred = float(arima["prediction"]) if arima.get("prediction") is not None else 0.0
        raw_prediction = (
            (lstm_w * lstm_pred)
            + (linear_w * linear_pred)
            + (arima_w * arima_pred)
        ) / weight_sum
        source = "ensemble-weighted"

    adjustment_bundle = _apply_intelligent_adjustments(raw_prediction, frame)
    next_day_prediction = adjustment_bundle["adjusted_prediction"]

    confidences = [
        value
        for value in [lstm.get("confidence"), linear.get("confidence"), arima.get("confidence")]
        if isinstance(value, (int, float))
    ]
    confidence_score = float(np.mean(confidences)) if confidences else 0.50
    confidence_score = max(0.05, min(0.95, confidence_score))

    # 95% confidence band width scaled by volatility and confidence.
    vol_index = adjustment_bundle["volatility_index"] or 50.0
    band_ratio = (0.03 + (vol_index / 1000.0)) * (1.0 - (0.4 * confidence_score))
    margin = next_day_prediction * band_ratio
    ci_low = next_day_prediction - margin
    ci_high = next_day_prediction + margin

    risk = _risk_from_volatility(adjustment_bundle["volatility_index"])
    recommendation = _recommendation(next_day_prediction, latest_price, risk)

    output = {
        "next_day_prediction": round(next_day_prediction, 2),
        "confidence_interval_95": {
            "low": round(ci_low, 2),
            "high": round(ci_high, 2),
        },
        "risk_assessment": risk,
        "recommendation": recommendation,
        "latest_btc_price": round(latest_price, 2),
        "confidence_score": round(confidence_score, 4),
        "prediction_source": source,
        "component_predictions": {
            "lstm": {
                "prediction": None if lstm["prediction"] is None else round(float(lstm["prediction"]), 2),
                "confidence": lstm["confidence"],
                "mse": lstm["mse"],
                "weight": lstm_w,
                "error": lstm.get("error"),
            },
            "linear": {
                "prediction": None if linear["prediction"] is None else round(float(linear["prediction"]), 2),
                "confidence": linear["confidence"],
                "mse": linear["mse"],
                "weight": linear_w,
                "caution": True,
                "error": linear.get("error"),
            },
            "arima": {
                "prediction": None if arima["prediction"] is None else round(float(arima["prediction"]), 2),
                "confidence": arima["confidence"],
                "mse": arima["mse"],
                "weight": arima_w,
                "is_valid": bool(arima.get("valid")),
                "error": arima.get("error"),
            },
        },
        "adjustments": adjustment_bundle["adjustments"],
        "volatility_index": adjustment_bundle["volatility_index"],
        "fear_greed_index": adjustment_bundle["fear_greed_index"],
    }

    if log_to_mlflow:
        mlflow.set_experiment("BTCUSD_Ensemble_Forecasting")
        with mlflow.start_run(run_name="BTCUSD_Ensemble_Next_Day"):
            mlflow.log_param("weight_lstm", MODEL_WEIGHTS["lstm"])
            mlflow.log_param("weight_arima", MODEL_WEIGHTS["arima"])
            mlflow.log_param("weight_linear", MODEL_WEIGHTS["linear"])
            mlflow.log_param("arima_mse_threshold", 1000)

            mlflow.log_metric("latest_btc_price", latest_price)
            mlflow.log_metric("ensemble_prediction", next_day_prediction)
            mlflow.log_metric("ensemble_ci_low", ci_low)
            mlflow.log_metric("ensemble_ci_high", ci_high)
            mlflow.log_metric("ensemble_confidence", confidence_score)
            mlflow.log_metric("volatility_index", float(vol_index))
            if adjustment_bundle["fear_greed_index"] is not None:
                mlflow.log_metric("fear_greed_index", float(adjustment_bundle["fear_greed_index"]))

            for model_key, payload in output["component_predictions"].items():
                if payload.get("prediction") is not None:
                    mlflow.log_metric(f"{model_key}_prediction", float(payload["prediction"]))
                if payload.get("confidence") is not None:
                    mlflow.log_metric(f"{model_key}_confidence", float(payload["confidence"]))
                if payload.get("mse") is not None:
                    mlflow.log_metric(f"{model_key}_mse", float(payload["mse"]))

    return output
