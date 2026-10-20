"""One-shot simplified GeoJSON for the app. Approximate boxes from catalog cities."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pjm_forecasting.areas import LOAD_AREAS

ROOT = Path(__file__).resolve().parent
SPLIT_ZONES = frozenset({"AE", "AEP", "ATSI", "DPL", "PEP", "PL"})
PAD_AREA = 0.35
PAD_ZONE = 0.25
PAD_REGION = 0.15


def _box(lons: list[float], lats: list[float], pad: float) -> list[list[float]]:
    min_lon, max_lon = min(lons) - pad, max(lons) + pad
    min_lat, max_lat = min(lats) - pad, max(lats) + pad
    return [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]


def _feature(feature_id: str, parent_id: str, level: str, ring: list[list[float]]) -> dict:
    return {
        "type": "Feature",
        "properties": {"id": feature_id, "parent_id": parent_id, "level": level},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _collection(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    by_zone: dict[str, list] = defaultdict(list)
    by_region: dict[str, list] = defaultdict(list)
    zone_region: dict[str, str] = {}
    for area in LOAD_AREAS:
        by_zone[area.zone].append(area)
        by_region[area.mkt_region].append(area)
        zone_region[area.zone] = area.mkt_region

    regions = [
        _feature(
            region,
            "PJM",
            "mkt_region",
            _box([a.lon for a in areas], [a.lat for a in areas], PAD_REGION),
        )
        for region, areas in sorted(by_region.items())
    ]
    zones = [
        _feature(
            zone,
            zone_region[zone],
            "zone",
            _box([a.lon for a in areas], [a.lat for a in areas], PAD_ZONE),
        )
        for zone, areas in sorted(by_zone.items())
    ]
    load_areas = [
        _feature(area.code, area.zone, "load_area", _box([area.lon], [area.lat], PAD_AREA))
        for area in LOAD_AREAS
        if area.zone in SPLIT_ZONES
    ]

    (ROOT / "regions.geojson").write_text(json.dumps(_collection(regions), separators=(",", ":")) + "\n")
    (ROOT / "zones.geojson").write_text(json.dumps(_collection(zones), separators=(",", ":")) + "\n")
    (ROOT / "load_areas.geojson").write_text(
        json.dumps(_collection(load_areas), separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
