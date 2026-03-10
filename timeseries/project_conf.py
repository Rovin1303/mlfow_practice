from pathlib import Path
import os
import sqlite3
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_BTCUSD_CSV = RAW_DATA_DIR / "btc-usd_historical.csv"
PROCESSED_BTCUSD_CSV = PROCESSED_DATA_DIR / "btcusd_processed.csv"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
DEFAULT_EXPERIMENTS = [
    "BTCUSD_Forecasting",
    "BTCUSD_Linear_Regression",
    "BTCUSD_ARIMA_Forecasting",
]
DEFAULT_REGISTERED_MODELS = [
    "BTCUSD_RNN_Model",
    "BTCUSD_Linear_Regression",
    "BTCUSD_ARIMA_Model",
]


def ensure_project_root_on_path():
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def configure_mlflow():
    _normalize_mlflow_artifact_locations()
    os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    return MLFLOW_TRACKING_URI


def _normalize_mlflow_artifact_locations():
    """Fix stale artifact locations copied from another machine/user.

    Some experiments were created with Unix-style paths such as `/Users/...`.
    On Windows this resolves to inaccessible locations and breaks model logging.
    """
    if not MLFLOW_DB_PATH.exists():
        return

    connection = sqlite3.connect(MLFLOW_DB_PATH)
    try:
        cursor = connection.cursor()
        rows = cursor.execute(
            """
            SELECT experiment_id, artifact_location
            FROM experiments
            WHERE artifact_location LIKE '/Users/%'
               OR artifact_location LIKE 'file:/Users/%'
               OR artifact_location LIKE 'file:///Users/%'
            """
        ).fetchall()

        for experiment_id, _ in rows:
            local_artifact_dir = PROJECT_ROOT / "mlruns" / str(experiment_id)
            local_artifact_dir.mkdir(parents=True, exist_ok=True)
            local_artifact_uri = f"file:///{local_artifact_dir.as_posix()}"
            cursor.execute(
                "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
                (local_artifact_uri, experiment_id),
            )

        if rows:
            connection.commit()
    finally:
        connection.close()
