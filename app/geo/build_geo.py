"""Regenerate ``pjm_regions.geojson`` from the load-area catalog.

The polygons are approximate territories, not utility boundaries. Each load
area gets one or more seed points placed by hand. Voronoi cells around the
seeds are clipped to a coarse PJM footprint, then dissolved by load area, zone,
and market region. That keeps the three levels tiled the same way, so a
region is exactly the union of its zones.

This is a developer tool. ``shapely`` is only needed to rerun it::

    PYTHONPATH=src uv run --no-project --with shapely python app/geo/build_geo.py

The checked-in GeoJSON is what the app serves.
"""

from __future__ import annotations

import json
from pathlib import Path

import shapely
from shapely.geometry import MultiPoint, Point, Polygon, mapping
from shapely.ops import unary_union

from pjm_forecasting.areas import BY_CODE, LOAD_AREAS

OUT = Path(__file__).with_name("pjm_regions.geojson")
GRID = 0.01  # degrees, roughly 1 km

# Coarse PJM footprint (lon, lat), clockwise from the Wisconsin line.
FOOTPRINT: list[tuple[float, float]] = [
    (-89.2, 42.45), (-87.8, 42.45), (-87.7, 42.0), (-87.55, 41.75), (-87.2, 41.62),
    (-86.85, 41.75), (-86.45, 42.12), (-85.6, 42.1), (-85.6, 41.75), (-84.8, 41.75),
    (-83.45, 41.72), (-82.9, 41.5), (-82.2, 41.5), (-81.7, 41.55), (-81.2, 41.78),
    (-80.55, 41.95), (-79.76, 42.27), (-79.76, 42.0), (-75.35, 42.0), (-75.1, 41.85),
    (-74.7, 41.35), (-74.0, 41.3), (-73.95, 41.1), (-73.92, 40.9), (-74.0, 40.7),
    (-74.0, 40.4), (-74.05, 40.0), (-74.1, 39.7), (-74.3, 39.35), (-74.9, 38.93),
    (-75.05, 38.45), (-75.2, 38.0), (-75.9, 37.15), (-76.05, 36.95), (-75.95, 36.55),
    (-76.2, 36.1), (-77.4, 35.95), (-78.2, 36.3), (-79.0, 36.54), (-81.7, 36.59),
    (-83.67, 36.6), (-84.6, 36.85), (-85.5, 36.8), (-85.4, 37.4), (-85.5, 38.3),
    (-85.4, 38.75), (-85.7, 39.2), (-85.6, 39.7), (-86.4, 40.4), (-87.0, 40.9),
    (-87.6, 41.0), (-88.4, 41.0), (-89.2, 41.2),
]  # fmt: skip

# Hand-placed seeds (lon, lat) per load area. More than one seed lets a
# territory spread out, the way AEP and DPL really do.
SEEDS: dict[str, list[tuple[float, float]]] = {
    # MIDATL
    "AECO": [(-74.5, 39.4), (-74.85, 39.1)],
    "VMEU": [(-75.05, 39.5)],
    "BC": [(-76.6, 39.3), (-76.9, 39.5)],
    "DPLCO": [(-75.55, 39.65), (-75.5, 39.1), (-75.35, 38.4), (-75.7, 37.8)],
    "EASTON": [(-76.0, 38.78)],
    "JC": [(-74.5, 40.8), (-74.2, 40.05), (-74.8, 40.95)],
    "ME": [(-75.9, 40.34), (-76.7, 40.0), (-76.4, 40.2)],
    "PE": [(-75.25, 40.0), (-75.6, 39.95)],
    "PEPCO": [(-77.0, 38.95), (-76.95, 39.12)],
    "SMECO": [(-76.8, 38.5), (-76.5, 38.3)],
    "PLCO": [(-75.5, 40.65), (-76.3, 40.7), (-76.85, 40.3)],
    "UGI": [(-75.9, 41.25)],
    "PN": [(-78.4, 40.5), (-79.0, 40.3), (-80.0, 41.9), (-78.7, 41.6), (-77.5, 41.7)],
    "PS": [(-74.2, 40.8), (-74.75, 40.2)],
    "RECO": [(-74.04, 41.11)],
    # WEST
    "AEPAPT": [(-79.94, 37.27), (-81.2, 37.4), (-82.0, 37.1), (-81.6, 38.35)],
    "AEPIMP": [(-85.14, 41.08), (-86.25, 41.68), (-85.4, 40.2), (-86.0, 40.8)],
    "AEPKPT": [(-82.7, 38.45), (-82.5, 37.6)],
    "AEPOPT": [(-83.0, 39.96), (-82.2, 39.5), (-81.6, 40.5), (-84.0, 40.7)],
    "AP": [(-79.96, 39.63), (-77.7, 39.64), (-78.76, 39.65), (-79.85, 38.9), (-79.55, 40.3)],
    "OE": [(-81.52, 41.08), (-81.7, 41.5), (-83.55, 41.65), (-82.55, 40.9), (-80.6, 41.15)],
    "PAPWR": [(-80.3, 40.95)],
    "CE": [(-87.63, 41.88), (-88.3, 41.8), (-89.0, 42.2), (-88.0, 41.2), (-89.0, 41.4)],
    "DAY": [(-84.19, 39.76), (-84.2, 40.25)],
    "DEOK": [(-84.5, 39.15), (-85.0, 39.3)],
    "DUQ": [(-80.0, 40.44), (-80.3, 40.6)],
    "EKPC": [(-84.5, 38.04), (-84.8, 37.3), (-84.2, 37.2), (-83.8, 37.8)],
    "OVEC": [(-85.38, 38.74), (-82.9, 39.0)],
    # SOUTH
    "DOM": [(-77.44, 37.54), (-76.3, 36.85), (-77.4, 38.9), (-78.5, 38.0), (-77.4, 36.2),
            (-77.5, 38.3), (-78.3, 38.9), (-78.5, 36.8)],
}  # fmt: skip


def _round(geom):
    return shapely.set_precision(geom, GRID)


def _cells() -> dict[str, list]:
    footprint = Polygon(FOOTPRINT)
    if not footprint.is_valid:
        raise RuntimeError("PJM footprint is not a simple polygon")
    labels: list[str] = []
    points: list[Point] = []
    for code, seeds in SEEDS.items():
        for lon, lat in seeds:
            point = Point(lon, lat)
            if not footprint.contains(point):
                raise RuntimeError(f"seed for {code} at {lon}, {lat} is outside the footprint")
            labels.append(code)
            points.append(point)
    unknown = set(SEEDS) - set(BY_CODE)
    missing = set(BY_CODE) - set(SEEDS)
    if unknown or missing:
        raise RuntimeError(f"seeds do not match catalog: unknown={unknown} missing={missing}")
    diagram = shapely.voronoi_polygons(MultiPoint(points), extend_to=footprint.buffer(5))
    by_label: dict[str, list] = {code: [] for code in SEEDS}
    for cell in diagram.geoms:
        for label, point in zip(labels, points, strict=True):
            if cell.contains(point):
                by_label[label].append(cell.intersection(footprint))
                break
        else:
            raise RuntimeError("Voronoi cell without a seed")
    return by_label


def _feature(feature_id: str, level: str, parent_id: str | None, name: str, geom) -> dict:
    return {
        "type": "Feature",
        "properties": {
            "id": feature_id,
            "parent_id": parent_id,
            "level": level,
            "name": name,
            "approximate": True,
        },
        "geometry": mapping(_round(geom)),
    }


def build() -> dict:
    by_label = _cells()
    area_geom = {code: unary_union(cells) for code, cells in by_label.items()}

    zone_areas: dict[str, list[str]] = {}
    zone_region: dict[str, str] = {}
    for area in LOAD_AREAS:
        zone_areas.setdefault(area.zone, []).append(area.code)
        zone_region[area.zone] = area.mkt_region
    zone_geom = {z: unary_union([area_geom[c] for c in codes]) for z, codes in zone_areas.items()}
    regions: dict[str, list[str]] = {}
    for zone, region in zone_region.items():
        regions.setdefault(region, []).append(zone)
    region_geom = {r: unary_union([zone_geom[z] for z in zs]) for r, zs in regions.items()}

    features = [_feature(r, "mkt_region", "PJM", r, g) for r, g in region_geom.items()]
    for zone, geom in zone_geom.items():
        features.append(_feature(zone, "zone", zone_region[zone], zone, geom))
    for zone, codes in zone_areas.items():
        if len(codes) > 1:  # only the six multi-area zones split into load areas
            features.extend(_feature(c, "load_area", zone, c, area_geom[c]) for c in codes)
    return {"type": "FeatureCollection", "features": features}


if __name__ == "__main__":
    collection = build()
    OUT.write_text(json.dumps(collection, separators=(",", ":")) + "\n")
    print(f"wrote {len(collection['features'])} features to {OUT} ({OUT.stat().st_size} bytes)")
