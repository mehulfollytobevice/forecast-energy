"""Smoke tests for the load-area package."""

import pjm_forecasting
from pjm_forecasting.areas import LOAD_AREAS
from pjm_forecasting.schema import WEATHER_COLUMNS
from pjm_forecasting.utils import load_config


def test_package_version():
    assert pjm_forecasting.__version__ == "0.1.0"


def test_default_config_matches_the_load_area_contract():
    config = load_config()
    assert config["data"]["history_start"] == "2018-01-01"
    assert config["horizon"]["prediction_length"] == 48
    assert config["weather"]["hourly"] == list(WEATHER_COLUMNS)
    assert config["load_areas"] == [area.code for area in LOAD_AREAS]
    assert config["model"]["horizon"] == "PT48H"
    assert config["model"]["quantiles"] == [0.1, 0.5, 0.9]
    assert config["model"]["country_code"] == "US"
    assert config["splits"]["train_end"] == "2025-01-01"
    assert config["splits"]["validation_end"] == "2026-01-01"
    assert config["storage"]["tracking_uri"] == "sqlite:///mlflow.db"
    assert config["forecast_runs"]["model"] == "ecmwf_ifs"
    assert config["forecast_runs"]["archive_start"] == "2024-03-14T00:00:00Z"
    assert config["backtest"]["start"] == "2026-01-01"
