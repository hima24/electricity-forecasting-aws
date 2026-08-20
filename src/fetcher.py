# fetcher.py — Live data retrieval
#
# Two data sources:
#   1. ENTSO-E Transparency Platform  →  load, generation, price
#   2. OpenWeatherMap                 →  weather for 5 Spanish cities

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from entsoe.entsoe import EntsoePandasClient

from config import (ENTSOE_API_KEY, OPENWEATHER_API_KEY,
                    COUNTRY_CODE, CITIES)


# Helpers

def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("h")


def _window(hours_back: int = 200):
    """Return (start, end) Timestamps covering the last N hours."""
    end   = _now_utc()
    start = end - pd.Timedelta(hours=hours_back)
    return start, end


# ENTSO-E

def fetch_entsoe_load(hours_back: int = 200) -> pd.DataFrame:
    """
    Fetch actual total load for Spain (or configured COUNTRY_CODE).
    Returns a DataFrame with columns [time, total_load_actual].
    """
    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
    start, end = _window(hours_back)

    try:
        raw = client.query_load(COUNTRY_CODE, start=start, end=end)
        # raw is a Series indexed by UTC timestamps
        df = raw.reset_index()
        df.columns = ["time", "total_load_actual"]
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.floor("h")
        df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        print(f"  [ENTSO-E] Load fetched: {len(df)} hours")
        return df
    except Exception as e:
        print(f"  [ENTSO-E] Load fetch error: {e}")
        return pd.DataFrame(columns=["time", "total_load_actual"])


def fetch_entsoe_generation(hours_back: int = 200) -> pd.DataFrame:
    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
    start, end = _window(hours_back)

    try:
        raw = client.query_generation(COUNTRY_CODE, start=start, end=end, psr_type=None)

        # Flatten MultiIndex columns — ENTSO-E returns (source, Actual/Forecast) tuples
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [
                "generation_" + str(a).lower().replace(" ", "_") + "_" + str(b).lower()
                if str(b) not in ("", "nan") else
                "generation_" + str(a).lower().replace(" ", "_")
                for a, b in raw.columns
            ]
        else:
            raw.columns = [
                "generation_" + str(c).lower().replace(" ", "_")
                for c in raw.columns
            ]

        raw = raw.reset_index()
        raw = raw.rename(columns={raw.columns[0]: "time"})
        raw["time"] = pd.to_datetime(raw["time"], utc=True).dt.floor("h")

        # Keep only the "actual" columns (drop forecast variants)
        actual_cols = ["time"] + [
            c for c in raw.columns
            if c != "time" and "forecast" not in c and "consumption" not in c
        ]
        raw = raw[actual_cols]

        # Standardise the key renewable column names
        rename_map = {}
        for col in raw.columns:
            if "solar" in col:
                rename_map[col] = "generation_solar"
            elif "wind_onshore" in col or ("wind" in col and "offshore" not in col):
                rename_map[col] = "generation_wind_onshore"
            elif "wind_offshore" in col or ("wind" in col and "offshore" in col):
                rename_map[col] = "generation_wind_offshore"
            elif "fossil_gas" in col or ("gas" in col and "fossil" in col):
                rename_map[col] = "generation_fossil_gas"
            elif "nuclear" in col:
                rename_map[col] = "generation_nuclear"
        raw = raw.rename(columns=rename_map)

        # Drop duplicate columns (keep first occurrence)
        raw = raw.loc[:, ~raw.columns.duplicated()]
        raw = raw.drop_duplicates("time").sort_values("time").reset_index(drop=True)

        print(f"  [ENTSO-E] Generation fetched: {len(raw)} hours, {raw.shape[1]-1} sources")
        return raw

    except Exception as e:
        print(f"  [ENTSO-E] Generation fetch error: {e}")
        return pd.DataFrame(columns=["time"])


def fetch_entsoe_prices(hours_back: int = 200) -> pd.DataFrame:
    """
    Fetch day-ahead electricity prices.
    Returns DataFrame with columns [time, price_day_ahead].
    """
    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
    start, end = _window(hours_back)

    try:
        raw = client.query_day_ahead_prices(COUNTRY_CODE, start=start, end=end)
        df = raw.reset_index()
        df.columns = ["time", "price_day_ahead"]
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.floor("h")
        df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        print(f"  [ENTSO-E] Prices fetched: {len(df)} hours")
        return df
    except Exception as e:
        print(f"  [ENTSO-E] Price fetch error: {e}")
        return pd.DataFrame(columns=["time", "price_day_ahead"])


def fetch_all_entsoe(hours_back: int = 200) -> pd.DataFrame:
    """Merge load, generation, and prices into one DataFrame."""
    load = fetch_entsoe_load(hours_back)
    gen  = fetch_entsoe_generation(hours_back)
    price = fetch_entsoe_prices(hours_back)

    df = load.copy()
    if not gen.empty:
        df = pd.merge_asof(df.sort_values("time"),
                           gen.sort_values("time"),
                           on="time", direction="nearest")
    if not price.empty:
        df = pd.merge_asof(df.sort_values("time"),
                           price.sort_values("time"),
                           on="time", direction="nearest")
    return df


# OpenWeatherMap
def fetch_weather_one_city(city: str) -> dict:
    """
    Fetch current weather for a single city.
    Returns a dict of weather variables.
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q":     city + ",ES",
        "appid": OPENWEATHER_API_KEY,
        "units": "standard",   # Kelvin, same as training data
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        d = resp.json()
        return {
            "temp":       d["main"]["temp"],
            "temp_min":   d["main"]["temp_min"],
            "temp_max":   d["main"]["temp_max"],
            "pressure":   d["main"]["pressure"],
            "humidity":   d["main"]["humidity"],
            "wind_speed": d["wind"]["speed"],
            "wind_deg":   d["wind"].get("deg", 0),
            "rain_1h":    d.get("rain", {}).get("1h", 0.0),
            "snow_3h":    d.get("snow", {}).get("3h", 0.0),
            "clouds_all": d["clouds"]["all"],
        }
    except Exception as e:
        print(f"  [OpenWeather] Error for {city}: {e}")
        return {}


def fetch_weather_all_cities() -> dict:
    """
    Fetch weather for all configured cities and return averaged values.
    Mirrors the training pipeline (5-city average).
    """
    all_rows = []
    for city in CITIES:
        row = fetch_weather_one_city(city)
        if row:
            all_rows.append(row)

    if not all_rows:
        print("  [OpenWeather] No city data retrieved.")
        return {}

    # Average numeric values across cities
    keys = all_rows[0].keys()
    averaged = {k: float(np.mean([r.get(k, np.nan) for r in all_rows])) for k in keys}
    print(f"  [OpenWeather] Weather averaged across {len(all_rows)}/{len(CITIES)} cities")
    return averaged


def fetch_weather_forecast_city(city: str, hours: int = 24) -> list[dict]:
    """
    Fetch 3-hourly weather forecast for one city (OWM free tier gives 5 days / 3h steps).
    Returns list of dicts, one per forecast step.
    """
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q":     city + ",ES",
        "appid": OPENWEATHER_API_KEY,
        "units": "standard",
        "cnt":   int(np.ceil(hours / 3)),
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json()["list"]
        rows = []
        for item in items:
            rows.append({
                "time":       pd.to_datetime(item["dt"], unit="s", utc=True).floor("h"),
                "temp":       item["main"]["temp"],
                "temp_min":   item["main"]["temp_min"],
                "temp_max":   item["main"]["temp_max"],
                "pressure":   item["main"]["pressure"],
                "humidity":   item["main"]["humidity"],
                "wind_speed": item["wind"]["speed"],
                "wind_deg":   item["wind"].get("deg", 0),
                "rain_1h":    item.get("rain", {}).get("3h", 0.0) / 3,
                "snow_3h":    item.get("snow", {}).get("3h", 0.0),
                "clouds_all": item["clouds"]["all"],
            })
        return rows
    except Exception as e:
        print(f"  [OpenWeather Forecast] Error for {city}: {e}")
        return []


def fetch_weather_forecast_averaged(hours: int = 24) -> pd.DataFrame:
    """
    Fetch and average weather forecasts across all cities.
    Returns DataFrame indexed by time with averaged weather columns.
    Uses forward-fill to interpolate 3h steps to hourly.
    """
    all_city_dfs = []
    for city in CITIES:
        rows = fetch_weather_forecast_city(city, hours)
        if rows:
            all_city_dfs.append(pd.DataFrame(rows).set_index("time"))

    if not all_city_dfs:
        return pd.DataFrame()

    # Average across cities at each timestamp
    combined = pd.concat(all_city_dfs).groupby(level=0).mean()

    # Reindex to hourly and forward-fill the 3h gaps
    now   = pd.Timestamp.now(tz="UTC").floor("h")
    end   = now + pd.Timedelta(hours=hours)
    idx   = pd.date_range(now, end, freq="h", tz="UTC")
    combined = combined.reindex(combined.index.union(idx)).sort_index().ffill().bfill()
    combined = combined.loc[idx]

    print(f"  [OpenWeather Forecast] {len(combined)} hourly steps averaged across {len(all_city_dfs)} cities")
    return combined.reset_index().rename(columns={"index": "time"})
