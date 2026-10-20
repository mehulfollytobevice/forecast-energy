"""Columns shared by the load cache, the weather cache, and the model frame."""

from __future__ import annotations

WEATHER_COLUMNS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "surface_pressure",
)

LOAD_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "timestamp_utc",
    "load_area",
    "zone",
    "mkt_region",
    "load",
)

FRAME_COLUMNS: tuple[str, ...] = LOAD_COLUMNS + WEATHER_COLUMNS
