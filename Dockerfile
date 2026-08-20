# Single image used for BOTH ECS services:
#   - the Streamlit dashboard (long-running web service)
#   - the hourly scheduler (run as a scheduled ECS task via EventBridge)
# Which one runs is decided by the container command at deploy time —
# see docs/AWS_DEPLOYMENT.md for the two task definitions.

FROM python:3.11-slim

WORKDIR /app

# System deps needed by lightgbm/xgboost wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

# Runtime data (DB, logs, selected_model.json) lives here — mount an EFS
# volume at this path in ECS so it survives task restarts/redeploys.
RUN mkdir -p /app/data
ENV DATA_DIR=/app/data
ENV MODELS_DIR=/app/models

WORKDIR /app/src

EXPOSE 8501

# Default: run the dashboard. The scheduler task definition overrides this
# command to `python scheduler.py` instead — see docs/AWS_DEPLOYMENT.md.
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
