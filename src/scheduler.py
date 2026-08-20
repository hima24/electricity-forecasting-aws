# scheduler.py - Hourly data fetch + forecast pipeline
#
# Run in a separate terminal alongside the dashboard:
#   python scheduler.py
#
# It will:
#   1. Immediately fetch data and run a forecast on startup
#   2. Read selected_model.json (written by the dashboard radio button)
#      to determine which model (LightGBM or XGBoost) to forecast with
#   3. Store predictions tagged with the model name so the dashboard
#      can display the correct results
#   4. Repeat every FETCH_INTERVAL_MIN minutes
#   5. Log everything to scheduler.log

import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler

from config import (
    MODEL_PATH, IMPUTER_PATH, FEATURES_PATH, PEAK_THRESHOLD_PATH,
    LIGHTGBM_MODEL_PATH, XGBOOST_MODEL_PATH, SELECTED_MODEL_PATH,
    FORECAST_HOURS, FETCH_INTERVAL_MIN, DATA_DIR,
)
from fetcher import fetch_all_entsoe, fetch_weather_forecast_averaged, fetch_weather_all_cities
from database import (init_db, upsert_actuals, upsert_predictions,
                      fill_in_errors, read_recent_for_features)
from features import build_live_row

# ---------------------------------------------------------------------------
# Model file paths  (train.py saves both of these after training)
# Resolved via config.py / env vars so this works the same locally,
# in Docker, and in ECS regardless of the container's working directory.
# ---------------------------------------------------------------------------
ALL_MODEL_PATHS = {
    "LightGBM": Path(LIGHTGBM_MODEL_PATH),
    "XGBoost":  Path(XGBOOST_MODEL_PATH),
}

# ---------------------------------------------------------------------------
# Shared selection file
# The dashboard writes this file whenever the user changes the radio button.
# The scheduler reads it at the start of every hourly job so it always uses
# whichever model the user has currently selected.
# ---------------------------------------------------------------------------
SELECTED_MODEL_FILE = Path(SELECTED_MODEL_PATH)
DEFAULT_MODEL       = "LightGBM"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "scheduler.log"),
        logging.StreamHandler(),  # also goes to stdout -> CloudWatch Logs in ECS
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared artefacts loaded once at startup (imputer / features / threshold).
# The active model itself is loaded fresh each cycle from the selection file.
# ---------------------------------------------------------------------------
log.info("Loading shared model artefacts ...")
try:
    imputer        = joblib.load(IMPUTER_PATH)
    feature_names  = joblib.load(FEATURES_PATH)
    peak_threshold = joblib.load(PEAK_THRESHOLD_PATH)
    log.info(
        f"  {len(feature_names)} features  |  "
        f"Peak threshold: {peak_threshold:,.0f} MW"
    )
except FileNotFoundError as e:
    log.error(f"Shared artefacts not found: {e}")
    log.error("Run  python train.py  first, then restart the scheduler.")
    raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_selected_model() -> str:
    """
    Return the model name currently selected in the dashboard radio button.
    Falls back to DEFAULT_MODEL if the file is missing or unreadable.
    """
    try:
        data = json.loads(SELECTED_MODEL_FILE.read_text(encoding="utf-8"))
        name = data.get("selected_model", DEFAULT_MODEL)
        if name in ALL_MODEL_PATHS:
            return name
        log.warning(
            f"Unknown model '{name}' in {SELECTED_MODEL_FILE} "
            f"- falling back to {DEFAULT_MODEL}."
        )
    except FileNotFoundError:
        pass  # file not written yet; use default silently
    except Exception as exc:
        log.warning(f"Could not read {SELECTED_MODEL_FILE}: {exc} - using {DEFAULT_MODEL}.")
    return DEFAULT_MODEL


def load_model_for_name(model_name: str):
    """
    Load and return the joblib model for the given name.
    Raises FileNotFoundError with a clear message if the .joblib is missing.
    """
    path = ALL_MODEL_PATHS[model_name]
    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} model file not found: {path}. "
            f"Run  python train.py  to generate it."
        )
    return joblib.load(path)


def store_predictions(fc_df: pd.DataFrame, model_name: str):
    """
    Persist forecast rows tagged with model_name so the dashboard
    can filter predictions by the selected model.
    """
    fc_df = fc_df.copy()
    fc_df["model_name"] = model_name
    try:
        upsert_predictions(fc_df, peak_threshold, model_name=model_name)
    except TypeError:
        # Older database.py without model_name kwarg - relies on the column.
        upsert_predictions(fc_df, peak_threshold)


# ---------------------------------------------------------------------------
# Core hourly job
# ---------------------------------------------------------------------------

def hourly_job():
    log.info("=" * 60)
    log.info("Hourly job started")

    # ------------------------------------------------------------------
    # 0. Load ALL available models.
    #    Both LightGBM and XGBoost are forecasted every cycle so their
    #    predictions are always in the DB. The dashboard radio button
    #    controls which one is *displayed* — not which one is run here.
    # ------------------------------------------------------------------
    active_models = {}
    for model_name, model_path in ALL_MODEL_PATHS.items():
        try:
            active_models[model_name] = load_model_for_name(model_name)
            log.info(f"  Loaded {model_name} from {model_path}")
        except FileNotFoundError as exc:
            log.warning(str(exc))

    if not active_models:
        log.error("No model files found at all. Run  python train.py  first.")
        return

    selected_model_name = read_selected_model()
    log.info(f"Dashboard selection: {selected_model_name}  (will forecast with all {len(active_models)} loaded models)")

    # ------------------------------------------------------------------
    # 1. Fetch latest actuals from ENTSO-E
    # ------------------------------------------------------------------
    log.info("Fetching ENTSO-E data ...")
    try:
        entsoe_df = fetch_all_entsoe(hours_back=200)
        if entsoe_df.empty:
            log.warning("  No ENTSO-E data returned - skipping this cycle.")
            return

        log.info("Fetching current weather ...")
        weather_now = fetch_weather_all_cities() or {}
        if weather_now:
            for k, v in weather_now.items():
                entsoe_df[k] = entsoe_df.get(k, v)

        ren_cols = [c for c in ["generation_wind_onshore", "generation_solar",
                                "generation_wind_offshore"] if c in entsoe_df.columns]
        if ren_cols:
            entsoe_df["renewable_gen"] = entsoe_df[ren_cols].fillna(0).sum(axis=1)
            if "total_load_actual" in entsoe_df.columns:
                load = entsoe_df["total_load_actual"].ffill().fillna(1)
                entsoe_df["renewable_ratio"] = entsoe_df["renewable_gen"] / (load + 1)

            solar_val = entsoe_df["generation_solar"].iloc[-1] \
                if "generation_solar" in entsoe_df.columns else np.nan
            wind_val  = entsoe_df["generation_wind_onshore"].iloc[-1] \
                if "generation_wind_onshore" in entsoe_df.columns else np.nan
            log.info(f"  [Renewables] Solar: {solar_val:,.0f} MW  |  Wind: {wind_val:,.0f} MW")
        else:
            log.info("  [Renewables] No renewable columns found in ENTSO-E response.")

        upsert_actuals(entsoe_df)
        log.info(f"  Actuals stored: {len(entsoe_df)} rows")

    except Exception as exc:
        log.error(f"ENTSO-E fetch failed: {exc}", exc_info=True)
        return

    # ------------------------------------------------------------------
    # 2. Fill in past forecast errors
    # ------------------------------------------------------------------
    try:
        fill_in_errors()
    except Exception as exc:
        log.warning(f"Error fill-in failed: {exc}")

    # ------------------------------------------------------------------
    # 3. Fetch weather forecast for the next N hours
    # ------------------------------------------------------------------
    log.info("Fetching weather forecast ...")
    try:
        weather_fc = fetch_weather_forecast_averaged(hours=FORECAST_HOURS)
    except Exception as exc:
        log.warning(f"Weather forecast failed: {exc}")
        weather_fc = pd.DataFrame()

    # ------------------------------------------------------------------
    # 4. Build feature rows and forecast with EVERY loaded model.
    #    Both are saved to the DB each cycle. The dashboard radio only
    #    controls which model's rows are *displayed* — not which runs.
    # ------------------------------------------------------------------
    log.info(f"Generating {FORECAST_HOURS}-hour forecasts for: {list(active_models.keys())} ...")
    base_history_df = read_recent_for_features(hours_back=200)

    if base_history_df.empty or len(base_history_df) < 24:
        log.warning("  Not enough history for lag features - skipping forecast.")
        return

    now = pd.Timestamp.now(tz="UTC").floor("h")

    for model_name, model in active_models.items():
        log.info(f"  Forecasting with {model_name} ...")

        # Each model gets its own independent rolling history copy
        history_df    = base_history_df.copy()
        forecast_rows = []

        for h in range(1, FORECAST_HOURS + 1):
            target_time = now + pd.Timedelta(hours=h)

            if not weather_fc.empty and "time" in weather_fc.columns:
                wf_row       = weather_fc[weather_fc["time"] == target_time]
                weather_dict = (wf_row.drop(columns=["time"]).iloc[0].to_dict()
                                if not wf_row.empty else {})
            else:
                weather_dict = weather_now or {}

            gen_cols = [c for c in history_df.columns
                        if c.startswith("generation_") and c in feature_names]
            gen_dict = history_df[gen_cols].iloc[-1].to_dict() if gen_cols else {}

            try:
                X_row = build_live_row(
                    history_df=history_df,
                    weather_row=weather_dict,
                    generation_row=gen_dict,
                    target_horizon_time=target_time,
                    feature_names=feature_names,
                    imputer=imputer,
                )

                pred = float(model.predict(X_row)[0])

                new_row = history_df.iloc[-1].copy()
                new_row["time"]              = target_time
                new_row["total_load_actual"] = pred
                history_df = pd.concat(
                    [history_df, pd.DataFrame([new_row])],
                    ignore_index=True,
                )

                forecast_rows.append({
                    "time":           target_time,
                    "predicted_load": pred,
                    "model_name":     model_name,
                })

            except Exception as exc:
                log.warning(f"  {model_name} forecast failed for {target_time}: {exc}")

        if forecast_rows:
            fc_df      = pd.DataFrame(forecast_rows)
            store_predictions(fc_df, model_name)
            peak_count = (fc_df["predicted_load"] >= peak_threshold).sum()
            log.info(
                f"  Stored {len(fc_df)} {model_name} rows  |  "
                f"Peak hours: {peak_count}"
            )
        else:
            log.warning(f"  No predictions generated for {model_name}.")

    log.info(f"Hourly job complete. Dashboard is showing: {selected_model_name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    init_db()

    # Create the default selection file if not present so the dashboard
    # radio always has something to read on first launch.
    if not SELECTED_MODEL_FILE.exists():
        SELECTED_MODEL_FILE.write_text(
            json.dumps({"selected_model": DEFAULT_MODEL}, indent=2),
            encoding="utf-8",
        )
        log.info(f"Created default selection file -> {SELECTED_MODEL_FILE}")

    log.info(f"Starting scheduler (interval={FETCH_INTERVAL_MIN} min) ...")
    hourly_job()

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        hourly_job,
        trigger="interval",
        minutes=FETCH_INTERVAL_MIN,
        id="hourly_fetch",
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped.")


if __name__ == "__main__":
    main()