"""PJM load-area demand forecasting.

Pipeline: Data Miner load and Open-Meteo weather, joined per load area, then
one OpenSTEF model per load area.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

_EXPORTS = {
    "prepare_openstef_frame": "pjm_forecasting.preprocessing",
    "split_chronological": "pjm_forecasting.splits",
    "ChronologicalSplit": "pjm_forecasting.splits",
    "train_area": "pjm_forecasting.training",
    "refit_area": "pjm_forecasting.training",
    "load_area_model": "pjm_forecasting.training",
    "predict_quantiles": "pjm_forecasting.training",
    "score_forecast": "pjm_forecasting.training",
    "run_area_backtest": "pjm_forecasting.backtesting",
    "select_forecast_versions": "pjm_forecasting.backtesting",
    "train_load_areas": "pjm_forecasting.flows.training",
    "train_aggregate_areas": "pjm_forecasting.flows.aggregates",
    "backtest_load_areas": "pjm_forecasting.flows.backtesting",
    "build_forecast_snapshot": "pjm_forecasting.snapshot",
    "write_forecast_snapshot": "pjm_forecasting.snapshot",
    "build_backtest_snapshot": "pjm_forecasting.backtest_snapshot",
    "write_backtest_snapshot": "pjm_forecasting.backtest_snapshot",
    "resolve_backtest_snapshot": "pjm_forecasting.backtest_snapshot",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_EXPORTS.keys()])
