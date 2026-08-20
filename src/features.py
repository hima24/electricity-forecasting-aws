# features.py — Shared feature engineering
#
# Used by both train.py (historical) and the live pipeline
# (scheduler.py) so the exact same features are always produced.


import numpy as np
import pandas as pd
import holidays

SPAIN_HOLS = holidays.Spain()


def engineer_features(df: pd.DataFrame, target: str = "total_load_actual") -> pd.DataFrame:
    """
    Add all engineered features to df in-place and return it.
    Assumes df already has a 'time' column (UTC datetime) and
    numeric columns for energy generation and weather variables.
    """
    df = df.copy()

    # Calendar
    df["hour"]        = df["time"].dt.hour
    df["day"]         = df["time"].dt.dayofweek
    df["month"]       = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear
    df["is_weekend"]  = df["day"].isin([5, 6]).astype(int)
    df["is_holiday"]  = df["time"].dt.date.apply(lambda x: int(x in SPAIN_HOLS))

    # Cyclic encodings — prevents the model treating hour 23 and 0 as far apart
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features (only when target column exists)
    if target in df.columns:
        df["lag_1"]      = df[target].shift(1)
        df["lag_24"]     = df[target].shift(24)
        df["lag_168"]    = df[target].shift(168)
        df["rolling_3"]  = df[target].shift(1).rolling(3).mean()
        df["rolling_24"] = df[target].shift(1).rolling(24).mean()
        df = df.dropna(
            subset=["lag_1", "lag_24", "rolling_3", "rolling_24"]
        ).reset_index(drop=True)

    # Renewable derived features
    ren = [
        c for c in ["generation_wind_onshore", "generation_solar",
                     "generation_wind_offshore"]
        if c in df.columns
    ]
    if ren:
        df["renewable_gen"] = df[ren].sum(axis=1)
        if target in df.columns:
            df["renewable_ratio"] = df["renewable_gen"] / (df[target] + 1)

    # Weather derived 
    if "temp_max" in df.columns and "temp_min" in df.columns:
        df["temp_range"] = df["temp_max"] - df["temp_min"]

    return df


def build_live_row(
    history_df: pd.DataFrame,
    weather_row: dict,
    generation_row: dict,
    target_horizon_time,
    feature_names: list,
    imputer,
) -> pd.DataFrame:
    """
    Build a single feature row for live forecasting.

    Parameters
    ----------
    history_df      : DataFrame of recent actuals (at least 168 rows)
    weather_row     : dict of weather variables for the target hour
    generation_row  : dict of generation variables for the target hour
    target_horizon_time : pd.Timestamp of the hour being forecast
    feature_names   : list of feature names from training
    imputer         : fitted SimpleImputer

    Returns
    -------
    X_row : DataFrame (1 row) ready to pass to model.predict()
    """
    t = target_horizon_time

    # Calendar features
    row = {
        "hour":        t.hour,
        "day":         t.dayofweek,
        "month":       t.month,
        "day_of_year": t.dayofyear,
        "is_weekend":  int(t.dayofweek in [5, 6]),
        "is_holiday":  int(t.date() in SPAIN_HOLS),
        "hour_sin":    np.sin(2 * np.pi * t.hour  / 24),
        "hour_cos":    np.cos(2 * np.pi * t.hour  / 24),
        "month_sin":   np.sin(2 * np.pi * t.month / 12),
        "month_cos":   np.cos(2 * np.pi * t.month / 12),
    }

    # Lag features from recent history
    recent = history_df["total_load_actual"].dropna().values
    row["lag_1"]      = recent[-1]   if len(recent) >= 1   else np.nan
    row["lag_24"]     = recent[-24]  if len(recent) >= 24  else np.nan
    row["lag_168"]    = recent[-168] if len(recent) >= 168 else np.nan
    row["rolling_3"]  = float(np.mean(recent[-3:]))  if len(recent) >= 3  else np.nan
    row["rolling_24"] = float(np.mean(recent[-24:])) if len(recent) >= 24 else np.nan

    # Weather
    row.update(weather_row)

    # Generation
    row.update(generation_row)

    # Renewable
    ren_val = sum(generation_row.get(k, 0) or 0
                  for k in ["generation_wind_onshore", "generation_solar",
                             "generation_wind_offshore"])
    row["renewable_gen"] = ren_val

    # Align to training feature names
    X = pd.DataFrame([row])
    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = np.nan
    X = X[feature_names]

    # Impute any missing
    X = pd.DataFrame(imputer.transform(X.astype(float)), columns=feature_names)
    return X
