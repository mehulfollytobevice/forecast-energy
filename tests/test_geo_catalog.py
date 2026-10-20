"""GeoJSON feature ids must match the load-area catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pjm_forecasting.areas import BY_CODE, LOAD_AREAS
from pjm_forecasting.reconcile import forecast_nodes

GEO = Path(__file__).resolve().parents[1] / "app" / "geo" / "pjm_regions.geojson"

MULTI_AREA_ZONES = {"AE", "AEP", "ATSI", "DPL", "PEP", "PL"}


@pytest.fixture(scope="module")
def features() -> list[dict]:
    assert GEO.is_file(), f"missing {GEO}; run app/geo/build_geo.py"
    collection = json.loads(GEO.read_text())
    assert collection["type"] == "FeatureCollection"
    return collection["features"]


def test_every_feature_has_catalog_fields(features: list[dict]) -> None:
    nodes = {(n.id, n.level): n for n in forecast_nodes()}
    for feature in features:
        props = feature["properties"]
        for key in ("id", "parent_id", "level", "name"):
            assert key in props
        node_key = (props["id"], props["level"])
        assert node_key in nodes, f"unknown feature {props['id']} level {props['level']}"
        node = nodes[node_key]
        assert props["parent_id"] == node.parent_id
        assert props["name"] == node.name
        assert feature["geometry"]["type"] in {"Polygon", "MultiPolygon"}


def test_split_zones_expose_load_areas(features: list[dict]) -> None:
    by_level: dict[str, set[str]] = {}
    for feature in features:
        level = feature["properties"]["level"]
        by_level.setdefault(level, set()).add(feature["properties"]["id"])
    zones = {area.zone for area in LOAD_AREAS}
    assert by_level["mkt_region"] == {"MIDATL", "WEST", "SOUTH"}
    assert by_level["zone"] == zones
    load_areas_in_geo = by_level.get("load_area", set())
    expected = {area.code for area in LOAD_AREAS if area.zone in MULTI_AREA_ZONES}
    assert load_areas_in_geo == expected
    assert load_areas_in_geo <= set(BY_CODE)
