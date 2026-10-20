# Notebooks

Exploratory and walkthrough notebooks for the **PJM regional load forecasting** pipeline. The production path is the package under `src/pjm_forecasting/` and the CLI scripts in `src/` — nothing the live forecast depends on should exist only here.

| Notebook | Purpose |
| --- | --- |
| [`01_eda_model_frame.ipynb`](01_eda_model_frame.ipynb) | Load/weather caches, distributions, join into the model frame |
| [`02_openstef_backtesting.ipynb`](02_openstef_backtesting.ipynb) | Reload a stored model, score a short holdout, then run an OpenSTEF Beam backtest on point-in-time ECMWF forecast runs |
| [`03_model_training.ipynb`](03_model_training.ipynb) | Prepare → split → fit → refit → holdout for one load area (knobs you can tweak without overwriting production MLflow) |

**Weather note:** EDA / chronological training often uses Open-Meteo archive (reanalysis). Beam backtests and live models use ECMWF IFS HRES single runs with publication and load-availability delays so evaluation does not leak future weather.
