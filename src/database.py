# database.py — SQLite storage for live actuals and predictions
#
# Three tables:
#   actuals     — real load, generation, weather fetched each hour
#   predictions — model forecasts made for each future hour (keyed by time + model_name)
#   errors      — difference between prediction and actual (filled in retroactively)

import sqlite3
import pandas as pd
import numpy as np
from config import DB_PATH


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist yet."""
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS actuals (
            time                        TEXT PRIMARY KEY,
            total_load_actual           REAL,
            price_day_ahead             REAL,
            generation_solar            REAL,
            generation_wind_onshore     REAL,
            generation_wind_offshore    REAL,
            generation_fossil_gas       REAL,
            generation_nuclear          REAL,
            temp                        REAL,
            temp_min                    REAL,
            temp_max                    REAL,
            humidity                    REAL,
            pressure                    REAL,
            wind_speed                  REAL,
            wind_deg                    REAL,
            rain_1h                     REAL,
            snow_3h                     REAL,
            clouds_all                  REAL,
            renewable_gen               REAL,
            renewable_ratio             REAL,
            fetched_at                  TEXT
        )
    """)

    # model_name is part of the PRIMARY KEY so LightGBM and XGBoost
    # predictions for the same hour are stored as separate rows.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            target_time     TEXT,
            model_name      TEXT,
            predicted_load  REAL,
            is_peak         INTEGER,
            made_at         TEXT,
            PRIMARY KEY (target_time, model_name)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS errors (
            time            TEXT,
            model_name      TEXT,
            actual          REAL,
            predicted       REAL,
            abs_error       REAL,
            pct_error       REAL,
            PRIMARY KEY (time, model_name)
        )
    """)

    # ---------------------------------------------------------------------------
    # Schema migration — rebuild predictions and errors tables if they are
    # missing ANY required column (made_at, model_name) or have the wrong
    # PRIMARY KEY.  We check the exact required column set so that even a
    # half-migrated table (model_name added via ALTER but made_at still
    # missing) is correctly detected and rebuilt.
    # ---------------------------------------------------------------------------
    PRED_REQUIRED = {"target_time", "model_name", "predicted_load", "is_peak", "made_at"}
    ERR_REQUIRED  = {"time", "model_name", "actual", "predicted", "abs_error", "pct_error"}

    # --- predictions ---
    pred_cols = {r[1] for r in cur.execute("PRAGMA table_info(predictions)").fetchall()}
    if pred_cols and not PRED_REQUIRED.issubset(pred_cols):
        print(f"  [DB] predictions schema mismatch (have: {pred_cols}, need: {PRED_REQUIRED}). Rebuilding ...")
        cur.execute("ALTER TABLE predictions RENAME TO predictions_old")
        cur.execute("""
            CREATE TABLE predictions (
                target_time     TEXT,
                model_name      TEXT,
                predicted_load  REAL,
                is_peak         INTEGER,
                made_at         TEXT,
                PRIMARY KEY (target_time, model_name)
            )
        """)
        old_cols = {r[1] for r in cur.execute("PRAGMA table_info(predictions_old)").fetchall()}
        # Build SELECT safely — only copy columns that exist in the old table
        mn_expr      = "model_name"      if "model_name" in old_cols else "'XGBoost'"
        made_at_expr = "made_at"         if "made_at"    in old_cols else "NULL"
        is_peak_expr = "is_peak"         if "is_peak"    in old_cols else "0"
        cur.execute(f"""
            INSERT OR IGNORE INTO predictions
                (target_time, model_name, predicted_load, is_peak, made_at)
            SELECT target_time, {mn_expr}, predicted_load, {is_peak_expr}, {made_at_expr}
            FROM predictions_old
        """)
        cur.execute("DROP TABLE predictions_old")
        print("  [DB] predictions table rebuilt successfully.")

    # --- errors ---
    err_cols = {r[1] for r in cur.execute("PRAGMA table_info(errors)").fetchall()}
    if err_cols and not ERR_REQUIRED.issubset(err_cols):
        print(f"  [DB] errors schema mismatch (have: {err_cols}, need: {ERR_REQUIRED}). Rebuilding ...")
        cur.execute("ALTER TABLE errors RENAME TO errors_old")
        cur.execute("""
            CREATE TABLE errors (
                time            TEXT,
                model_name      TEXT,
                actual          REAL,
                predicted       REAL,
                abs_error       REAL,
                pct_error       REAL,
                PRIMARY KEY (time, model_name)
            )
        """)
        old_cols = {r[1] for r in cur.execute("PRAGMA table_info(errors_old)").fetchall()}
        mn_expr = "model_name" if "model_name" in old_cols else "'XGBoost'"
        cur.execute(f"""
            INSERT OR IGNORE INTO errors
                (time, model_name, actual, predicted, abs_error, pct_error)
            SELECT time, {mn_expr}, actual, predicted, abs_error, pct_error
            FROM errors_old
        """)
        cur.execute("DROP TABLE errors_old")
        print("  [DB] errors table rebuilt successfully.")

    conn.commit()
    conn.close()
    print("  [DB] Tables ready.")


def upsert_actuals(df: pd.DataFrame):
    """Insert or replace rows in the actuals table."""
    conn = get_connection()
    now  = pd.Timestamp.now(tz="UTC").isoformat()

    for _, row in df.iterrows():
        time_str = pd.Timestamp(row["time"]).isoformat()
        conn.execute("""
            INSERT OR REPLACE INTO actuals
                (time, total_load_actual, price_day_ahead,
                 generation_solar, generation_wind_onshore, generation_wind_offshore,
                 generation_fossil_gas, generation_nuclear,
                 temp, temp_min, temp_max, humidity, pressure,
                 wind_speed, wind_deg, rain_1h, snow_3h, clouds_all,
                 renewable_gen, renewable_ratio, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            time_str,
            _g(row, "total_load_actual"),
            _g(row, "price_day_ahead"),
            _g(row, "generation_solar"),
            _g(row, "generation_wind_onshore"),
            _g(row, "generation_wind_offshore"),
            _g(row, "generation_fossil_gas"),
            _g(row, "generation_nuclear"),
            _g(row, "temp"),
            _g(row, "temp_min"),
            _g(row, "temp_max"),
            _g(row, "humidity"),
            _g(row, "pressure"),
            _g(row, "wind_speed"),
            _g(row, "wind_deg"),
            _g(row, "rain_1h"),
            _g(row, "snow_3h"),
            _g(row, "clouds_all"),
            _g(row, "renewable_gen"),
            _g(row, "renewable_ratio"),
            now,
        ))
    conn.commit()
    conn.close()


def upsert_predictions(forecast_df: pd.DataFrame, peak_threshold: float, model_name: str = "XGBoost"):
    """
    Save forecast rows keyed by (target_time, model_name).
    LightGBM and XGBoost predictions for the same hour are stored as
    separate rows so the dashboard can filter by model independently.

    forecast_df must have columns: [time, predicted_load]
    model_name: "LightGBM" or "XGBoost"
    """
    conn = get_connection()
    now  = pd.Timestamp.now(tz="UTC").isoformat()

    for _, row in forecast_df.iterrows():
        time_str = pd.Timestamp(row["time"]).isoformat()
        is_peak  = int(float(row["predicted_load"]) >= peak_threshold)
        # Use model_name from the row if present, otherwise use the argument
        mn = row.get("model_name", model_name) if hasattr(row, "get") else model_name
        conn.execute("""
            INSERT OR REPLACE INTO predictions
                (target_time, model_name, predicted_load, is_peak, made_at)
            VALUES (?, ?, ?, ?, ?)
        """, (time_str, str(mn), float(row["predicted_load"]), is_peak, now))

    conn.commit()
    conn.close()


def fill_in_errors():
    """
    For any prediction whose target_time has now passed and has an actual,
    compute and store the error — separately per model_name.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT p.target_time, p.model_name, p.predicted_load, a.total_load_actual
        FROM predictions p
        JOIN actuals a ON p.target_time = a.time
        WHERE a.total_load_actual IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM errors e
              WHERE e.time = p.target_time AND e.model_name = p.model_name
          )
    """).fetchall()

    for r in rows:
        actual    = r["total_load_actual"]
        predicted = r["predicted_load"]
        abs_err   = abs(actual - predicted)
        pct_err   = abs_err / (actual + 1e-6) * 100
        conn.execute("""
            INSERT OR REPLACE INTO errors
                (time, model_name, actual, predicted, abs_error, pct_error)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (r["target_time"], r["model_name"], actual, predicted, abs_err, pct_err))

    conn.commit()
    conn.close()
    if rows:
        print(f"  [DB] Filled in errors for {len(rows)} prediction rows.")


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def read_actuals(hours_back: int = 200) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(f"""
        SELECT * FROM actuals
        ORDER BY time DESC
        LIMIT {hours_back}
    """, conn)
    conn.close()
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def read_predictions(hours_ahead: int = 24) -> pd.DataFrame:
    """
    Return future predictions for ALL models so the dashboard can
    filter by the selected model_name after loading.
    """
    conn  = get_connection()
    now   = pd.Timestamp.now(tz="UTC").isoformat()
    limit = (hours_ahead + 1) * 2   # x2 because we store two models per hour
    df = pd.read_sql(f"""
        SELECT target_time, model_name, predicted_load, is_peak, made_at
        FROM predictions
        WHERE target_time >= '{now}'
        ORDER BY model_name, target_time
        LIMIT {limit}
    """, conn)
    conn.close()
    if df.empty:
        return df
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
    # Rename model_name -> model so the dashboard normalize function finds it
    df = df.rename(columns={"model_name": "model"})
    return df


def read_errors(days_back: int = 7) -> pd.DataFrame:
    """Return errors for all models; dashboard can group/filter as needed."""
    conn  = get_connection()
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)).isoformat()
    df = pd.read_sql(f"""
        SELECT * FROM errors
        WHERE time >= '{since}'
        ORDER BY time
    """, conn)
    conn.close()
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    # Rename model_name -> model for consistency with read_predictions
    if "model_name" in df.columns and "model" not in df.columns:
        df = df.rename(columns={"model_name": "model"})
    return df


def read_recent_for_features(hours_back: int = 200) -> pd.DataFrame:
    """Read recent actuals for lag/rolling feature construction."""
    return read_actuals(hours_back)


# ---------------------------------------------------------------------------
# Internal util
# ---------------------------------------------------------------------------

def _g(row, col):
    """Safe getter — returns None if column missing or value is NaN."""
    val = row.get(col, None)
    if val is None:
        return None
    try:
        v = float(val)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None