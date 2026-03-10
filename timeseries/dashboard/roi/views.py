from datetime import timedelta
from pathlib import Path
import math
import sys

from django.shortcuts import render
from django.utils import timezone
from mlflow.tracking import MlflowClient
import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_conf import MLFLOW_TRACKING_URI, PROCESSED_BTCUSD_CSV
from .models import ROIMetric


MODEL_SOURCES = [
    {
        "label": "RNN",
        "experiment": "BTCUSD_Forecasting",
        "prediction_keys": ["rnn_next_day_pred", "next_day_pred"],
    },
    {
        "label": "Linear",
        "experiment": "BTCUSD_Linear_Regression",
        "prediction_keys": ["linear_next_day_pred", "next_day_pred"],
    },
    {
        "label": "ARIMA",
        "experiment": "BTCUSD_ARIMA_Forecasting",
        "prediction_keys": ["arima_next_day_pred", "next_day_pred"],
    },
]
TRADE_ENTRY_THRESHOLD = 0.002  # 0.2%


def _pick_first_numeric(metrics, keys):
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _read_processed_data():
    if not PROCESSED_BTCUSD_CSV.exists():
        return None
    try:
        return pd.read_csv(PROCESSED_BTCUSD_CSV)
    except Exception:
        return None


def _get_latest_finished_run(client, experiment_name):
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def _collect_latest_forecasts(client):
    forecasts = []
    for source in MODEL_SOURCES:
        run = _get_latest_finished_run(client, source["experiment"])
        if not run:
            continue
        metrics = run.data.metrics or {}
        prediction = _pick_first_numeric(metrics, source["prediction_keys"])
        if prediction is None:
            continue
        forecasts.append(
            {
                "model": source["label"],
                "prediction": prediction,
                "run_id": run.info.run_id[:8],
                "accuracy_pct": _pick_first_numeric(metrics, ["accuracy_pct"]),
            }
        )
    return forecasts


def _max_drawdown(equity_curve):
    peak = None
    max_dd = 0.0
    for value in equity_curve:
        if peak is None or value > peak:
            peak = value
        if peak and peak > 0:
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _simulate_roi(df, forecasts, lookback_days=30, starting_capital=10000.0):
    if df is None or df.empty:
        return None
    if "Close" not in df.columns or "Daily_Return" not in df.columns:
        return None
    if len(df) < lookback_days + 2:
        return None

    frame = df.tail(lookback_days + 1).copy()
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame["Daily_Return"] = pd.to_numeric(frame["Daily_Return"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["Close"])

    if len(frame) < lookback_days + 1:
        return None

    latest_close = float(frame["Close"].iloc[-1])

    if forecasts:
        best = min(forecasts, key=lambda item: abs(item["prediction"] - latest_close))
        selected_model = best["model"]
        selected_prediction = float(best["prediction"])
    else:
        selected_model = "NoForecast"
        selected_prediction = latest_close

    implied_move = (selected_prediction - latest_close) / latest_close if latest_close > 0 else 0.0

    if "MA7" in frame.columns and "MA21" in frame.columns:
        ma7 = pd.to_numeric(frame["MA7"], errors="coerce").fillna(method="ffill").fillna(frame["Close"])
        ma21 = pd.to_numeric(frame["MA21"], errors="coerce").fillna(method="ffill").fillna(frame["Close"])
        trend_signal = ma7 > ma21
    else:
        trend_signal = frame["Daily_Return"].rolling(3).mean().fillna(0.0) > 0

    strategy_returns = []
    baseline_returns = []
    trade_returns = []

    for i in range(len(frame) - 1):
        next_day_return = float(frame["Daily_Return"].iloc[i + 1])
        baseline_returns.append(next_day_return)

        take_trade = implied_move > TRADE_ENTRY_THRESHOLD and bool(trend_signal.iloc[i])
        r = next_day_return if take_trade else 0.0
        strategy_returns.append(r)
        if take_trade:
            trade_returns.append(next_day_return)

    strategy_equity = [starting_capital]
    baseline_equity = [starting_capital]

    for r in strategy_returns:
        strategy_equity.append(strategy_equity[-1] * (1.0 + r))
    for r in baseline_returns:
        baseline_equity.append(baseline_equity[-1] * (1.0 + r))

    strategy_profit = strategy_equity[-1] - starting_capital
    strategy_dd = _max_drawdown(strategy_equity)
    baseline_dd = _max_drawdown(baseline_equity)

    if baseline_dd > 0:
        risk_reduction_pct = max(0.0, ((baseline_dd - strategy_dd) / baseline_dd) * 100.0)
    else:
        risk_reduction_pct = 0.0

    win_rate_pct = (sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100.0) if trade_returns else 0.0

    return {
        "model_version": selected_model,
        "period": f"Last {lookback_days} Days",
        "simulated_profit_usd": float(strategy_profit),
        "risk_reduction_pct": float(risk_reduction_pct),
        "latest_close": latest_close,
        "selected_prediction": selected_prediction,
        "implied_move_pct": float(implied_move * 100.0),
        "trade_count": int(len(trade_returns)),
        "win_rate_pct": float(win_rate_pct),
        "entry_threshold_pct": float(TRADE_ENTRY_THRESHOLD * 100.0),
    }


def _should_persist_snapshot(latest_roi, snapshot):
    if latest_roi is None:
        return True
    if timezone.now() - latest_roi.calculated_at > timedelta(hours=6):
        return True
    if latest_roi.model_version != snapshot["model_version"]:
        return True
    if abs(float(latest_roi.simulated_profit_usd) - float(snapshot["simulated_profit_usd"])) >= 5.0:
        return True
    if abs(float(latest_roi.risk_reduction_pct) - float(snapshot["risk_reduction_pct"])) >= 0.1:
        return True
    return False


def roi_index(request):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    processed_df = _read_processed_data()
    forecasts = _collect_latest_forecasts(client)
    snapshot = _simulate_roi(processed_df, forecasts)

    latest_roi = ROIMetric.objects.order_by("-calculated_at").first()

    if snapshot:
        if _should_persist_snapshot(latest_roi, snapshot):
            latest_roi = ROIMetric.objects.create(
                model_version=snapshot["model_version"],
                period=snapshot["period"],
                simulated_profit_usd=snapshot["simulated_profit_usd"],
                risk_reduction_pct=snapshot["risk_reduction_pct"],
            )
    elif latest_roi is None:
        latest_roi = ROIMetric.objects.create(
            model_version="N/A",
            period="Last 30 Days",
            simulated_profit_usd=0.0,
            risk_reduction_pct=0.0,
        )
        snapshot = {
            "latest_close": None,
            "selected_prediction": None,
            "implied_move_pct": 0.0,
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "entry_threshold_pct": float(TRADE_ENTRY_THRESHOLD * 100.0),
        }

    context = {
        "latest_roi": latest_roi,
        "history": ROIMetric.objects.all().order_by("-calculated_at")[:10],
        "strategy": snapshot,
        "forecasts": forecasts,
    }
    return render(request, "dashboard/roi_index.html", context)