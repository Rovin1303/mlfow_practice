import argparse
import os
import subprocess
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from project_conf import PROCESSED_BTCUSD_CSV, configure_mlflow


PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD_ROOT = PROJECT_ROOT / "dashboard"
EXPERIMENTS = (
    "BTCUSD_Forecasting",
    "BTCUSD_Linear_Regression",
    "BTCUSD_ARIMA_Forecasting",
)
TRAINING_SCRIPTS = (
    "src/data/ingestion.py",
    "src/models/train.py",
    "src/models/linear_regression.py",
    "src/models/arima_model.py",
)
MODEL_TRAINING_STEPS = (
    ("BTCUSD_Forecasting", "src/models/train.py"),
    ("BTCUSD_Linear_Regression", "src/models/linear_regression.py"),
    ("BTCUSD_ARIMA_Forecasting", "src/models/arima_model.py"),
)


def run_step(args, cwd=PROJECT_ROOT, env=None):
    print(f"\n==> Running (cwd={cwd}): {' '.join(args)}")
    subprocess.run(args, cwd=str(cwd), env=env, check=True)


def _safe_registered_model_versions_count(client: MlflowClient) -> int:
    try:
        total = 0
        for rm in client.search_registered_models():
            versions = getattr(rm, "latest_versions", None) or []
            total += len(versions)
        return total
    except Exception:
        return 0


def get_mlflow_snapshot(tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    snapshot = {
        "tracking_uri": tracking_uri,
        "experiments": [],
        "total_runs": 0,
        "registered_versions": _safe_registered_model_versions_count(client),
    }

    for name in EXPERIMENTS:
        exp = client.get_experiment_by_name(name)
        if not exp:
            snapshot["experiments"].append(
                {"name": name, "exists": False, "runs": 0, "best_mse": None}
            )
            continue

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=1000,
            order_by=["start_time DESC"],
        )

        best_mse = None
        for r in runs:
            mse = r.data.metrics.get("mse")
            if mse is None:
                continue
            if best_mse is None or mse < best_mse:
                best_mse = mse

        snapshot["total_runs"] += len(runs)
        snapshot["experiments"].append(
            {
                "name": name,
                "exists": True,
                "runs": len(runs),
                "best_mse": best_mse,
            }
        )

    return snapshot


def has_any_runs(tracking_uri: str) -> bool:
    snapshot = get_mlflow_snapshot(tracking_uri)
    return snapshot["total_runs"] > 0


def _has_finished_runs(client: MlflowClient, experiment_name: str) -> bool:
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return False

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return bool(runs)


def _run_incremental_training(env):
    run_step([sys.executable, "src/data/ingestion.py"], cwd=PROJECT_ROOT, env=env)

    mlflow.set_tracking_uri(env["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    for experiment_name, script in MODEL_TRAINING_STEPS:
        if _has_finished_runs(client, experiment_name):
            print(f"Skipping {script}: finished run exists for {experiment_name}.")
            continue
        run_step([sys.executable, script], cwd=PROJECT_ROOT, env=env)


def print_snapshot(snapshot):
    print("\n========== Pipeline Diagnostics ==========")
    print(f"Project root           : {PROJECT_ROOT}")
    print(f"Dashboard root         : {DASHBOARD_ROOT}")
    print(f"MLFLOW_TRACKING_URI    : {snapshot['tracking_uri']}")
    print(
        f"Processed data exists  : {PROCESSED_BTCUSD_CSV.exists()} -> {PROCESSED_BTCUSD_CSV}"
    )
    print(f"Registered versions    : {snapshot['registered_versions']}")
    print(f"Total runs             : {snapshot['total_runs']}")
    print("Experiments:")
    for e in snapshot["experiments"]:
        if not e["exists"]:
            print(f"  - {e['name']}: NOT FOUND")
        else:
            print(f"  - {e['name']}: runs={e['runs']}, best_mse={e['best_mse']}")
    print("==========================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap and run the BTCUSD forecasting dashboard."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8000")
    parser.add_argument(
        "--mode",
        choices=["serve", "incremental", "full"],
        default="serve",
        help=(
            "Pipeline mode: serve (fast startup), incremental (ingest + train only missing models), "
            "or full (ingest + retrain all models)."
        ),
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print full MLflow diagnostics before and after pipeline execution.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Compatibility flag: equivalent to --mode serve.",
    )
    args = parser.parse_args()

    if args.skip_train:
        args.mode = "serve"

    tracking_uri = configure_mlflow()
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = tracking_uri

    # Ensure imports resolve for both Django app packages and project modules.
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(DASHBOARD_ROOT), str(PROJECT_ROOT), existing_pp]
    ).strip(os.pathsep)

    if args.diagnostics:
        print_snapshot(get_mlflow_snapshot(tracking_uri))

    # Run Django from dashboard root so `monitoring.urls` resolves.
    run_step([sys.executable, "manage.py", "migrate"], cwd=DASHBOARD_ROOT, env=env)

    if args.mode == "full":
        for script in TRAINING_SCRIPTS:
            run_step([sys.executable, script], cwd=PROJECT_ROOT, env=env)
    elif args.mode == "incremental":
        _run_incremental_training(env)
    elif not PROCESSED_BTCUSD_CSV.exists():
        # Keep minimum bootstrap for fast startup mode.
        run_step([sys.executable, "src/data/ingestion.py"], cwd=PROJECT_ROOT, env=env)

    if args.diagnostics:
        print_snapshot(get_mlflow_snapshot(tracking_uri))

    run_step(
        [sys.executable, "manage.py", "runserver", f"{args.host}:{args.port}"],
        cwd=DASHBOARD_ROOT,
        env=env,
    )


if __name__ == "__main__":
    main()