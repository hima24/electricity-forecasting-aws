# train.py — Train and compare Linear Regression, Random Forest, and XGBoost
#
# Run this ONCE before starting the live dashboard:
#   python train.py
#
# Requires:
#   energy_dataset.csv
#   weather_features.csv
#   config.py
#   features.py

import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb

from config import (
    MODEL_PATH,
    IMPUTER_PATH,
    FEATURES_PATH,
    PEAK_THRESHOLD_PATH,
)

from features import engineer_features


def load_historical():
    print("Loading historical CSVs ...")

    energy = pd.read_csv("energy_dataset.csv")
    weather = pd.read_csv("weather_features.csv")

    energy["time"] = pd.to_datetime(
        energy["time"], utc=True, errors="coerce"
    ).dt.floor("h")

    weather["dt_iso"] = pd.to_datetime(
        weather["dt_iso"], utc=True, errors="coerce"
    ).dt.floor("h")

    energy = energy.dropna(subset=["time"]).copy()
    weather = weather.dropna(subset=["dt_iso"]).copy()

    for col in energy.columns:
        if energy[col].dtype == object or str(energy[col].dtype).startswith("string"):
            energy[col] = pd.to_numeric(energy[col], errors="coerce")

    for col in weather.columns:
        if weather[col].dtype == object or str(weather[col].dtype).startswith("string"):
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    numeric_weather_cols = [
        col for col in weather.columns
        if pd.api.types.is_numeric_dtype(weather[col]) and col != "dt_iso"
    ]

    weather_avg = (
        weather.groupby("dt_iso")[numeric_weather_cols]
        .mean()
        .reset_index()
    )

    energy = energy.sort_values("time").reset_index(drop=True)
    weather_avg = weather_avg.sort_values("dt_iso").reset_index(drop=True)

    df = pd.merge_asof(
        energy,
        weather_avg,
        left_on="time",
        right_on="dt_iso",
        direction="nearest"
    )

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df = df.sort_values("time").reset_index(drop=True).ffill()

    print(f"  Merged dataset: {df.shape[0]:,} rows")
    return df


def build_feature_matrix(df, target="total_load_actual"):
    """
    Apply feature engineering and return X, y, feature_names, imputer.
    """

    df = engineer_features(df, target=target)

    exclude = {
        "time",
        "dt_iso",
        "weather_main",
        "weather_description",
        "weather_icon",
        "weather_id",
        "city_name",
        "total_load_forecast",
        "forecast_solar_day_ahead",
        "forecast_wind_onshore_day_ahead",
        "forecast_wind_offshore_eday_ahead",
        target,
        "price_actual",
        "price_day_ahead",
        "is_peak",
        "ren_bin",
        "date_only",
    }

    features = [
        col for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]

    X_raw = df[features].copy()

    empty_cols = X_raw.columns[X_raw.isna().all()].tolist()
    if empty_cols:
        X_raw = X_raw.drop(columns=empty_cols)

    features = list(X_raw.columns)

    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(
        imputer.fit_transform(X_raw.astype(float)),
        columns=features
    )

    y = df[target].reset_index(drop=True)

    return X, y, features, imputer


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    print(f"\nTraining {model_name} ...")

    try:
        if model_name == "XGBoost":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
        elif model_name == "LightGBM":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
        else:
            model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae  = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2   = r2_score(y_test, predictions)

        print(f"  {model_name} done  |  RMSE: {rmse:,.2f}  MAE: {mae:,.2f}  R2: {r2:.4f}")

        return {
            "Model":         model_name,
            "MAE":           mae,
            "RMSE":          rmse,
            "R2":            r2,
            "Trained_Model": model,
        }

    except Exception as exc:
        print(f"  ERROR training {model_name}: {exc}")
        return None


def main():
    df = load_historical()

    print("\nBuilding feature matrix ...")
    X, y, features, imputer = build_feature_matrix(df)

    print(f"  Features used: {len(features)}")

    split = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"  Training rows: {len(X_train):,}")
    print(f"  Testing rows : {len(X_test):,}")

    threshold = float(y_train.quantile(0.90))
    joblib.dump(threshold, PEAK_THRESHOLD_PATH)
    print(f"\nPeak threshold saved: {threshold:,.0f} MW")

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
            n_jobs=-1
        ),

         "LightGBM": lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
        ),
    }

    results = []

    for model_name, model in models.items():
        result = evaluate_model(
            model_name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )
        if result is not None:
            results.append(result)
        else:
            print(f"  SKIPPED {model_name} — training failed, it will not appear in comparison or be saved.")

    results_df = pd.DataFrame([
        {
            "Model": r["Model"],
            "MAE": r["MAE"],
            "RMSE": r["RMSE"],
            "R2": r["R2"],
        }
        for r in results
    ])

    results_df = results_df.sort_values(by="RMSE", ascending=True)

    print("\nModel Comparison:")
    print(results_df.to_string(index=False, formatters={
        "MAE": "{:,.2f}".format,
        "RMSE": "{:,.2f}".format,
        "R2": "{:.4f}".format,
    }))

    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nSaved model comparison table -> model_comparison_results.csv")

    best_model_name = results_df.iloc[0]["Model"]
    best_result = next(r for r in results if r["Model"] == best_model_name)
    best_model = best_result["Trained_Model"]

    print(f"\nBest Model: {best_model_name}")
    print(f"Best RMSE : {best_result['RMSE']:,.2f} MW")

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    joblib.dump(features, FEATURES_PATH)

    print(f"\nSaved best model -> {MODEL_PATH}")
    print(f"Saved imputer    -> {IMPUTER_PATH}")
    print(f"Saved features   -> {FEATURES_PATH}")

    # Save LightGBM and XGBoost separately so the scheduler can load
    # whichever one the user selects in the dashboard radio button.
    # Both files MUST exist — raise immediately if either is missing
    # so the user knows to re-check before starting the scheduler.
    lgbm_result = next((r for r in results if r["Model"] == "LightGBM"), None)
    xgb_result  = next((r for r in results if r["Model"] == "XGBoost"),  None)

    missing = []

    if lgbm_result:
        joblib.dump(lgbm_result["Trained_Model"], "lightgbm_model.joblib")
        print(f"Saved lightgbm_model.joblib  (RMSE: {lgbm_result['RMSE']:,.2f} MW)")
    else:
        missing.append("LightGBM")
        print("ERROR: LightGBM did not train successfully — lightgbm_model.joblib NOT saved.")

    if xgb_result:
        joblib.dump(xgb_result["Trained_Model"], "xgboost_model.joblib")
        print(f"Saved xgboost_model.joblib   (RMSE: {xgb_result['RMSE']:,.2f} MW)")
    else:
        missing.append("XGBoost")
        print("ERROR: XGBoost did not train successfully — xgboost_model.joblib NOT saved.")

    if missing:
        raise RuntimeError(
            f"Training incomplete — the following models failed and their .joblib files "
            f"were not saved: {missing}. Check the error messages above, fix the issue, "
            f"and re-run train.py before starting the scheduler."
        )

    print("\nTraining complete.")
    print("You can now run:")
    print("  python scheduler.py")
    print("  streamlit run dashboard.py")


if __name__ == "__main__":
    main()