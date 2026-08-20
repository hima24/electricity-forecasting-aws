# ⚡ Electricity Demand Forecasting — Spain

Hourly electricity demand forecasting for Spain's national grid, with a live
Streamlit dashboard, an hourly data-collection scheduler, and a model
comparison pipeline (Linear Regression, Random Forest, XGBoost, LightGBM).

Built for **COSC/INDE 6397 — Big Data Analytics** (Team 6: Himavarsha,
Lyba Siddiqui, Ranjitha Basavaraju).

## Results

| Model | RMSE (MW) | MAE (MW) | R² |
|---|---|---|---|
| **XGBoost** | **507.5** | — | **0.9874** |
| LightGBM | 493.3† | — | — |
| Random Forest | — | — | — |
| Linear Regression | — | — | — |

† From a later retraining run — see `docs/model_comparison_results.csv` for
the exact numbers produced by the model in this repo.

- 24-hour-ahead hourly load forecasting
- Peak-demand classification (top decile of load): **F1 = 0.901**
- Confirmed the **Merit Order Effect** on Spain's day-ahead price data
- Deployed as a live dashboard with an **hourly scheduler** and **SQLite**
  backend, refetching real grid + weather data every hour and re-forecasting
  the next 24 hours

## Architecture

```
┌─────────────────┐      hourly       ┌──────────────┐
│  ENTSO-E API     │ ───────────────▶ │              │
│  (grid load,     │                  │  scheduler.py │──▶ live_data.db (SQLite)
│  generation,     │                  │  (APScheduler)│      actuals / predictions / errors
│  day-ahead price)│                  │              │
└─────────────────┘                  └──────┬───────┘
┌─────────────────┐                         │ loads xgboost_model.joblib
│ OpenWeatherMap   │ ───────────────────────▶│ or lightgbm_model.joblib
│ (5-city weather) │                         │ (user-selected)
└─────────────────┘                         ▼
                                    ┌──────────────────┐
                                    │  dashboard.py     │◀── reads live_data.db
                                    │  (Streamlit UI)   │
                                    └──────────────────┘
```

Two long-running processes share one SQLite database:
- **`scheduler.py`** — fetches fresh actuals every `FETCH_INTERVAL_MIN`
  minutes, engineers features, forecasts the next `FORECAST_HOURS` hours with
  **every** trained model, and stores results tagged by model name.
- **`dashboard.py`** — a Streamlit app that reads from the same database and
  lets the user toggle between the LightGBM and XGBoost forecasts.

## Repo structure

```
├── src/
│   ├── config.py          # env-var-based settings — no secrets committed
│   ├── fetcher.py         # ENTSO-E + OpenWeatherMap API clients
│   ├── features.py        # shared feature engineering (train + live)
│   ├── database.py        # SQLite schema + read/write helpers
│   ├── train.py           # trains & compares all 4 models, saves artifacts
│   ├── scheduler.py       # hourly fetch → feature → forecast → store loop
│   └── dashboard.py       # Streamlit live dashboard
├── models/                 # trained model artifacts (see note below)
├── docs/
│   ├── model_comparison_results.csv
│   └── AWS_DEPLOYMENT.md  # step-by-step cloud deployment guide
├── requirements.txt
├── Dockerfile
├── .env.example
└── .gitignore
```

> **Note on `models/`:** this repo ships the trained `xgboost_model.joblib`
> and `lightgbm_model.joblib` used by the live scheduler/dashboard, plus
> `imputer.pkl` and `peak_threshold.pkl`. The generic best-model bundle
> (`model.pkl`) and the saved feature-name list (`features.pkl`) that
> `train.py` also produces aren't included yet — run `python src/train.py`
> once locally (with the two CSVs below) to regenerate them before starting
> the scheduler, or add your existing copies into `models/`.

## Data

Trained on the [Hourly Energy Demand Generation and Weather — Spain
dataset](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)
(Kaggle). The raw CSVs (`energy_dataset.csv`, `weather_features.csv`, ~26MB
combined) aren't committed to this repo — download them from Kaggle and
place them at the repo root before running `train.py`.

## Local setup

```bash
git clone <this-repo>
cd electricity-demand-forecasting
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your real ENTSOE_API_KEY and OPENWEATHER_API_KEY
#   - ENTSO-E key: https://transparency.entsoe.eu/ (free, request via email)
#   - OpenWeatherMap key: https://openweathermap.org/api (free tier)

# 1. Train (only needed once, or to retrain)
cd src
python train.py

# 2. Start the hourly scheduler (separate terminal)
python scheduler.py

# 3. Start the dashboard (separate terminal)
streamlit run dashboard.py
```

## Docker

```bash
docker build -t electricity-forecast .
docker run --env-file .env -p 8501:8501 electricity-forecast
```

Runs the dashboard by default. To run the scheduler instead:

```bash
docker run --env-file .env electricity-forecast python scheduler.py
```

## Cloud deployment (AWS)

Deployed on **AWS ECS Fargate** (dashboard as a long-running service,
scheduler as an EventBridge-triggered scheduled task), with credentials in
**AWS Secrets Manager** and the SQLite database on an **EFS** volume so it
survives redeploys. Full walkthrough with exact CLI commands:
**[`docs/AWS_DEPLOYMENT.md`](docs/AWS_DEPLOYMENT.md)**.

## Security note

No API keys are hardcoded anywhere in this codebase. All credentials are
read from environment variables via `src/config.py`
(`os.environ.get(...)`) — locally via a git-ignored `.env` file, and in
production via AWS Secrets Manager injected as ECS task environment
variables. `.gitignore` excludes `.env`, the SQLite database, logs, and the
raw training CSVs.

## Tech stack

Python · pandas · scikit-learn · XGBoost · LightGBM · Streamlit · Plotly ·
APScheduler · SQLite · Docker · AWS (ECS Fargate, EventBridge, Secrets
Manager, ECR, EFS)
