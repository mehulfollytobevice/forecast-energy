"""OpenSTEF configuration follows the load-area catalog. No model is fit."""

from __future__ import annotations

from datetime import timedelta

from pjm_forecasting.areas import BY_CODE
from pjm_forecasting.settings import load_settings
from pjm_forecasting.training import model_id_for_area, workflow_config_for_area


def test_workflow_config_uses_the_area_coordinate_and_horizon():
    settings = load_settings()
    area = BY_CODE["AECO"]
    config = workflow_config_for_area(area, settings, None, run_name="final", reuse=False)
    horizon = config.horizons[0].value
    assert model_id_for_area("aeco") == "pjm-AECO"
    assert config.model == "xgboost"
    assert config.model_id == "pjm-AECO"
    assert config.run_name == "final"
    assert str(config.location.country_code) == "US"
    assert config.location.name == "AECO"
    assert float(config.location.coordinate.latitude) == area.lat
    assert float(config.location.coordinate.longitude) == area.lon
    assert horizon == timedelta(hours=48)
    assert config.sample_interval == timedelta(hours=1)
    assert [float(quantile) for quantile in config.quantiles] == [0.1, 0.5, 0.9]
    assert config.mlflow_storage is None
    assert config.temperature_column == "temperature_2m"
    assert config.radiation_column == "shortwave_radiation"
