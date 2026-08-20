# config.py — Central configuration, loaded from environment variables.
#
# LOCAL DEV:
#   Create a .env file (copy .env.example) and this will be picked up
#   automatically via python-dotenv.
#
# CLOUD (AWS):
#   Real values are injected as environment variables by ECS from
#   AWS Secrets Manager — nothing sensitive ever lives in this file
#   or in the repo. See docs/AWS_DEPLOYMENT.md.

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; in prod, real env vars are already set

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent           # repo root
MODELS_DIR = Path(os.environ.get("MODELS_DIR", BASE_DIR / "models"))
DATA_DIR   = Path(os.environ.get("DATA_DIR",   BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH              = os.environ.get("DB_PATH",              str(DATA_DIR / "live_data.db"))
MODEL_PATH           = os.environ.get("MODEL_PATH",           str(MODELS_DIR / "model.pkl"))
IMPUTER_PATH         = os.environ.get("IMPUTER_PATH",         str(MODELS_DIR / "imputer.pkl"))
FEATURES_PATH        = os.environ.get("FEATURES_PATH",        str(MODELS_DIR / "features.pkl"))
PEAK_THRESHOLD_PATH  = os.environ.get("PEAK_THRESHOLD_PATH",  str(MODELS_DIR / "peak_threshold.pkl"))
LIGHTGBM_MODEL_PATH  = os.environ.get("LIGHTGBM_MODEL_PATH",  str(MODELS_DIR / "lightgbm_model.joblib"))
XGBOOST_MODEL_PATH   = os.environ.get("XGBOOST_MODEL_PATH",   str(MODELS_DIR / "xgboost_model.joblib"))
SELECTED_MODEL_PATH  = os.environ.get("SELECTED_MODEL_PATH",  str(DATA_DIR / "selected_model.json"))

# ---------------------------------------------------------------------------
# Credentials — REQUIRED at runtime, never hardcoded, never committed.
# Locally: set in .env (git-ignored). In AWS: injected from Secrets Manager.
# ---------------------------------------------------------------------------
ENTSOE_API_KEY      = os.environ.get("ENTSOE_API_KEY", "")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

if not ENTSOE_API_KEY or not OPENWEATHER_API_KEY:
    import warnings
    warnings.warn(
        "ENTSOE_API_KEY and/or OPENWEATHER_API_KEY are not set. "
        "Set them in a local .env file (see .env.example) or as environment "
        "variables / Secrets Manager entries in production. "
        "The dashboard will still run against existing data in the DB, "
        "but the scheduler's live fetch will fail without them."
    )

# ---------------------------------------------------------------------------
# App settings
# ---------------------------------------------------------------------------
COUNTRY_CODE        = os.environ.get("COUNTRY_CODE", "ES")
CITIES              = os.environ.get(
    "CITIES", "Madrid,Barcelona,Valencia,Seville,Bilbao"
).split(",")
FORECAST_HOURS      = int(os.environ.get("FORECAST_HOURS", 24))
FETCH_INTERVAL_MIN  = int(os.environ.get("FETCH_INTERVAL_MIN", 60))
