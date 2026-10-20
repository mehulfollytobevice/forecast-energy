# PJM Regional Load Forecasting

**Goal:** forecast hourly electricity demand for the entire [PJM](https://www.pjm.com/) Interconnection — the largest wholesale electricity market in North America — accurately enough to be useful for operations and planning.

PJM covers ~65 million people across 13 mid-Atlantic and Midwest states. Its load is the net of weather-driven demand, behind-the-meter solar, industrial schedules, and regional diversity. Getting the next 48 hours right is a core skill in renewable energy: it drives unit commitment, reserve procurement, storage dispatch, and how much variable wind and solar the grid can absorb without imbalance.

This repository is an end-to-end load-forecasting system: data ingestion → feature frames → quantile models → hierarchical reconciliation → live snapshots → an interactive map dashboard. It is built as a portfolio project to demonstrate production-minded forecasting practice, not a toy notebook demo.

---

## Why this problem

Renewable-heavy grids amplify forecast error. Under-forecast load and you short reserves or curtail renewables inefficiently; over-forecast and you run expensive peaking capacity. Hiring teams at ISOs, utilities, and renewables shops care whether you can:

- Define the **right grain** (here: metered distribution companies, not a single RTO series)
- Avoid **look-ahead leakage** in weather and load availability
- Produce **calibrated uncertainty**, not only a point forecast
- Roll forecasts up a **hierarchy** so zone and RTO totals stay coherent
- **Compare against a real baseline** (PJM’s own published forecast)

Those are the design choices this project is organized around.

---

## What I built

| Piece | What it does |
| --- | --- |
| **29 load-area models** | One hourly XGBoost quantile model (P10 / P50 / P90) per PJM distribution company, 48-hour horizon, via [OpenSTEF](https://github.com/OpenSTEF/openstef) |
| **Hierarchy** | Load areas → transmission zones → market regions (MIDATL / WEST / SOUTH) → full PJM. Displayed P50 at parents is the sum of children; uncertainty bands come from models of the totals |
| **Realistic weather** | Live training and backtests use ECMWF IFS HRES **single-run** archives (point-in-time), not reanalysis that would leak future weather |
| **Live pipeline** | Refresh metered + near-real-time load, pull the newest weather run, predict, widen bands, reconcile, write a JSON snapshot |
| **PJM benchmark** | Overlay PJM’s `load_frcstd_7_day` forecast on the dashboard for an apples-to-apples comparison |
| **Dashboard** | Map + fan chart for the live 48h forecast and a backtest tab (actual vs as-issued) |
| **Engineering** | Prefect flows, MLflow registry, conformal band calibration, quality gates, pytest suite, optional Railway / S3 deploy |

```text
PJM Data Miner ──► load areas (hourly MW)
Open-Meteo     ──► weather (archive | ECMWF single runs)
        │
        ▼
  joined frame → OpenSTEF XGBoost (per area) → quantiles
        │
        ▼
  hierarchy reconcile → forecast / backtest snapshot → FastAPI + map UI
                              │
                     PJM 7-day forecast (benchmark)
```

---

## Method (forecasting decisions that matter)

### Forecast grain

A **load area** is one metered distribution company in PJM’s `hrl_load_metered` feed. That is the modeling grain. Several transmission zones contain more than one company (for example AE = AECO + VMEU). Coarser totals are sums of `load` on the same hour — never PJM’s published `RTO` row, which is dropped so rollups are not double-counted.

Once models exist, a zone or full-PJM forecast is built from the load-area forecasts. Summing child quantile bands is wider than training a model on the total; parent P10/P90 therefore come from models of those aggregates, then are shifted onto the reconciled P50 sum.

### Weather without leakage

| Use | Weather source | Why |
| --- | --- | --- |
| Research / chronological split | Open-Meteo **archive** (reanalysis) | Dense history from 2018 for EDA and baseline fits |
| Live models & Beam backtests | ECMWF IFS HRES **single runs** | Only weather that would have been published by prediction time |

A run is treated as available `publication_delay_hours` after initialization (default 6h). Metered load for an hour is treated as known `load_availability_delay_hours` after the hour begins (default 1h). The live job fills PJM’s ~10-day metered lag with `inst_load` / `hrl_load_prelim`, split to load areas by trailing metered shares.

### Uncertainty

Models emit P10 / P50 / P90. Validation residuals drive **per-lead-hour conformal scales** that widen bands later in the 48-hour horizon. Scoring reports MAE, pinball loss, and empirical P10–P90 coverage.

### Evaluation posture

- Chronological train / validation / holdout (archive path)
- Rolling validation for forecast-run models (last 90 days held out)
- OpenSTEF **Beam** walk-forward backtests that retrain on a schedule and never reuse the stored final model for scoring
- **PJM’s official short-term load forecast** plotted next to ours on shared nodes (RTO, regions, zones). Each fetched vintage is saved so it can be scored later

---

## Stack

- **Python 3.12**, pandas / pyarrow, Prefect, MLflow  
- **OpenSTEF 4.4.8** (XGBoost quantile forecasting + Beam backtesting)  
- **PJM Data Miner** (metered load, prelim/instantaneous load, 7-day forecast)  
- **Open-Meteo** (archive + Single Runs API)  
- **FastAPI + React (Vite) + MapLibre** dashboard

---

## Repository layout

```text
src/pjm_forecasting/     # library: ingest, frames, training, reconcile, snapshots
src/*.py                 # CLI refresh scripts (load, weather, frames)
app/                     # FastAPI API + React map dashboard
configs/default.yaml     # horizons, splits, live schedule, quality gates
notebooks/               # EDA, training walkthrough, Beam backtest demos
tests/                   # unit + smoke tests (CI skips live network / full OpenSTEF fits)
data/                    # gitignored caches and snapshots (regenerate locally)
```

Catalog of the 29 load areas, zones, regions, and weather cities: `src/pjm_forecasting/areas.py`.

---

## Quick start

Python 3.12+. Training uses OpenSTEF with the Beam extra.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# set PJM_API_KEY (https://apiportal.pjm.com/profile)
# optional: OPEN_METEO_API_KEY for commercial Open-Meteo / Single Runs
```

### Build data and train one area

```bash
# Metered load (2018 → today) and archive weather
python src/update_zone_load.py
python src/update_zone_weather.py --start 2018-01-01 --end 2024-12-31
python src/build_load_area_frame.py

# Research path: chronological split, archive weather
python -c "from pjm_forecasting.flows.training import train_load_areas; train_load_areas(areas='AECO')"

# Live path: train on ECMWF single-run weather and publish locally
PYTHONPATH=src python -m pjm_forecasting.jobs.retrain --areas AECO
```

### Live 48-hour forecast

```bash
PYTHONPATH=src python -m pjm_forecasting.jobs.forecast --now
# or a quick dry run with cached data:
PYTHONPATH=src python -m pjm_forecasting.jobs.forecast --now --areas AECO --skip-download
```

### Dashboard

```bash
uv sync --extra dashboard --extra dev
uv run uvicorn app.api.main:app --reload --port 8000
cd app/web && npm install && npm run dev
```

Open `http://127.0.0.1:5173`. Tabs: **Live forecast** (next 48h) and **Backtest** (actual vs as-issued). Fixtures under `app/fixtures/` let the UI run without a full data pull.

### Tests

```bash
pytest -q -m "not openstef and not network" --cov=src --cov-report=term-missing
```

GitHub Actions runs that command on every push/PR. `pytest -m openstef` fits a small local model and is opt-in.

---

## Data contract

Training and inference rows share the same columns. OpenSTEF takes `timestamp` as the index and `load` as the target; it builds lags and calendar features itself.

| Column | Meaning |
| --- | --- |
| `timestamp` | US/Eastern hour beginning |
| `timestamp_utc` | Same hour in UTC |
| `load_area` / `zone` / `mkt_region` | Hierarchy keys |
| `load` | Metered MW (null on forecast hours) |
| `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `shortwave_radiation`, `surface_pressure` | Open-Meteo predictors |

History starts **2018-01-01**. Areas PJM had not yet published are absent until their first hour; rollups sum the areas that exist. Quantile columns are model outputs, not frame fields.

Zones that split into multiple load areas:

| Zone | Load areas |
| --- | --- |
| AE | AECO, VMEU |
| AEP | AEPAPT, AEPIMP, AEPKPT, AEPOPT |
| ATSI | OE, PAPWR |
| DPL | DPLCO, EASTON |
| PEP | PEPCO, SMECO |
| PL | PLCO, UGI |

Every other transmission zone is a single load area with the same code. UGI is stored as zone PL / MIDATL.

### On-disk layout (gitignored)

```text
data/raw/pjm/load_areas/{LOAD_AREA}/YYYY-MM.parquet
data/raw/openmeteo/load_areas/{LOAD_AREA}/...
data/raw/openmeteo/forecast_runs/{LOAD_AREA}/{YYYYMMDDTHHMMZ}.parquet
data/processed/load_area_hourly.parquet
data/processed/openstef/{LOAD_AREA}/{train,validation,test}.parquet
data/processed/forecast_runs/{LOAD_AREA}.parquet
data/processed/backtests/{LOAD_AREA}.parquet
data/processed/forecast_snapshot.json
data/processed/backtest_snapshot.json
```

Local MLflow defaults to `sqlite:///mlflow.db` and `artifacts/mlflow/`. Override with `MLFLOW_TRACKING_URI`.

---

## Pipeline reference

### Refresh

```bash
python src/update_zone_load.py
python src/update_zone_weather.py --start 2018-01-01 --end 2024-12-31
python src/build_load_area_frame.py

python src/update_zone_weather.py --forecast
python src/build_load_area_frame.py --inference
```

`--dry-run` on the load script lists missing months without calling Data Miner. `--areas AECO` limits either download.

### Backtest

```bash
python src/update_historical_weather_runs.py --start 2026-01-01 --end 2026-01-08 --areas AECO
python -c "from pjm_forecasting.flows.backtesting import backtest_load_areas; backtest_load_areas(areas='AECO')"
# reuse cache only:
python -c "from pjm_forecasting.flows.backtesting import backtest_load_areas; backtest_load_areas(areas='AECO', skip_download=True)"

PYTHONPATH=src python -c "from pjm_forecasting.backtest_snapshot import write_backtest_snapshot; print(write_backtest_snapshot())"
```

### Predict from a published model

```python
from pjm_forecasting import load_area_model, predict_quantiles

workflow = load_area_model("AECO")
# history: AECO rows from the joined frame, indexed by timestamp
forecast = predict_quantiles("AECO", history, origin="2026-01-15 06:00", workflow=workflow)
```

### Production-style dashboard bundle

```bash
cd app/web && npm run build
FORECAST_SNAPSHOT=app/fixtures/forecast_snapshot.json \
BACKTEST_SNAPSHOT=app/fixtures/backtest_snapshot.json \
uv run uvicorn app.api.main:app --port 8000
```

Then open `http://127.0.0.1:8000/`. Snapshot resolution for `/api/forecast`: object store `snapshots/forecast/latest.json` → `FORECAST_SNAPSHOT` → `data/processed/forecast_snapshot.json` → fixture.

---

## Notebooks

See [`notebooks/README.md`](notebooks/README.md) for EDA, a single-area training walkthrough, and Beam backtesting. Nothing the live forecast depends on lives only in a notebook.

---

## Deploy on Railway

The dashboard, hourly forecast job, and weekly retrain job are one Railway project defined in [`.railway/railway.ts`](.railway/railway.ts) (Infrastructure as Code — Config as Code `railway.toml` is deprecated).

| Service | Role |
| --- | --- |
| **web** | FastAPI + map UI (`Dockerfile` default command, healthcheck `/api/health`) |
| **forecast** | Cron `15 * * * *` UTC; runs `python -m pjm_forecasting.jobs.forecast` and exits |
| **retrain** | Cron `15 6 * * 1` UTC (Mondays); publishes a new model bundle |
| **snapshots** | S3-compatible bucket shared by all three services |

The forecast job still gates on Eastern `live.schedule_et` (`05:15`, `11:15`, `17:15`, `23:15`), so most hourly ticks exit with `skipped: schedule`.

```bash
npm install
railway login
railway link          # this GitHub repo / existing project
railway config plan   # preview: web + forecast + retrain + snapshots bucket
railway config apply  # creates/updates those resources — review the plan first
```

Then in the Railway dashboard:

1. On **web**, generate a public domain. Open `https://<domain>/api/health` — expect `{"status":"ok"}`.
2. Set `PJM_API_KEY` and `OPEN_METEO_API_KEY` on **web**, **forecast**, and **retrain** (or as shared variables). The IaC file uses `preserve()` so those values are not stored in git.
3. Bucket credentials are wired automatically as `S3_BUCKET`, `S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION`.
4. Publish models into the bucket (local `.env` with the same `S3_*` vars):
   ```bash
   PYTHONPATH=src python -m pjm_forecasting.jobs.retrain --bootstrap
   ```
5. Run one live forecast outside the Eastern schedule:
   ```bash
   PYTHONPATH=src python -m pjm_forecasting.jobs.forecast --now
   ```
   or temporarily set the **forecast** start command to that with `--now`, trigger a run, then restore the IaC start command.

If **web** already exists from a GitHub deploy, rename it to `web` before applying so IaC updates that service instead of creating a duplicate. Do not add `railway.toml` / `railway.json` — new services cannot opt into Config as Code.

---

## Status

Working locally end-to-end: ingest → train → live forecast job → dashboard. Hosted refresh is Railway IaC (`.railway/railway.ts`): web + forecast cron + retrain cron + object storage.

---

## License

MIT. PJM and Open-Meteo retain rights to their data — respect each provider’s terms of use.

---

*Built by [Mehul Jain](https://github.com/mehulfollytobevice)* 
