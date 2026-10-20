"""PJM load areas, the forecast grain, and the cities used for weather.

Transmission zone, market region, and full PJM are sums of these areas.
Coordinates are one city per area and are shared by the archive backfill
and the live forecast.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadArea:
    """One fully metered distribution company inside PJM."""

    code: str
    zone: str
    mkt_region: str
    city: str
    lat: float
    lon: float


# Zones that are a single load area keep the city already used for that zone.
# Split zones get a city inside each company. UGI is filed under PL / MIDATL
# even when Data Miner tags the row OTHER.
LOAD_AREAS: tuple[LoadArea, ...] = (
    LoadArea("AECO", "AE", "MIDATL", "Atlantic City", 39.3643, -74.4229),
    LoadArea("VMEU", "AE", "MIDATL", "Vineland", 39.4862, -75.0257),
    LoadArea("BC", "BC", "MIDATL", "Baltimore", 39.2904, -76.6122),
    LoadArea("DPLCO", "DPL", "MIDATL", "Salisbury", 38.3607, -75.5994),
    LoadArea("EASTON", "DPL", "MIDATL", "Easton", 38.7743, -76.0763),
    LoadArea("JC", "JC", "MIDATL", "Morristown", 40.7968, -74.4815),
    LoadArea("ME", "ME", "MIDATL", "Reading", 40.3356, -75.9269),
    LoadArea("PE", "PE", "MIDATL", "Philadelphia", 39.9526, -75.1652),
    LoadArea("PEPCO", "PEP", "MIDATL", "Washington", 38.9072, -77.0369),
    LoadArea("SMECO", "PEP", "MIDATL", "Hughesville", 38.5318, -76.7836),
    LoadArea("PLCO", "PL", "MIDATL", "Allentown", 40.6084, -75.4902),
    LoadArea("UGI", "PL", "MIDATL", "Kingston", 41.2618, -75.8969),
    LoadArea("PN", "PN", "MIDATL", "Altoona", 40.5187, -78.3947),
    LoadArea("PS", "PS", "MIDATL", "Newark", 40.7357, -74.1724),
    LoadArea("RECO", "RECO", "MIDATL", "Spring Valley", 41.1132, -74.0438),
    LoadArea("AEPAPT", "AEP", "WEST", "Roanoke", 37.2710, -79.9414),
    LoadArea("AEPIMP", "AEP", "WEST", "Fort Wayne", 41.0793, -85.1394),
    LoadArea("AEPKPT", "AEP", "WEST", "Ashland", 38.4784, -82.6379),
    LoadArea("AEPOPT", "AEP", "WEST", "Columbus", 39.9612, -82.9988),
    LoadArea("AP", "AP", "WEST", "Morgantown", 39.6295, -79.9559),
    LoadArea("OE", "ATSI", "WEST", "Akron", 41.0814, -81.5190),
    LoadArea("PAPWR", "ATSI", "WEST", "New Castle", 41.0037, -80.3470),
    LoadArea("CE", "CE", "WEST", "Chicago", 41.8781, -87.6298),
    LoadArea("DAY", "DAY", "WEST", "Dayton", 39.7589, -84.1916),
    LoadArea("DEOK", "DEOK", "WEST", "Cincinnati", 39.1031, -84.5120),
    LoadArea("DUQ", "DUQ", "WEST", "Pittsburgh", 40.4406, -79.9959),
    LoadArea("EKPC", "EKPC", "WEST", "Lexington", 38.0406, -84.5037),
    LoadArea("OVEC", "OVEC", "WEST", "Madison", 38.7359, -85.3797),
    LoadArea("DOM", "DOM", "SOUTH", "Richmond", 37.5407, -77.4360),
)

if len({area.code for area in LOAD_AREAS}) != len(LOAD_AREAS):
    raise RuntimeError("LOAD_AREAS has duplicate load-area codes")

BY_CODE: dict[str, LoadArea] = {area.code: area for area in LOAD_AREAS}
CATALOG_CODES: frozenset[str] = frozenset(BY_CODE)


def resolve_areas(spec: str) -> list[str]:
    """``all`` (or blank) selects every load area, in catalog order."""
    text = spec.strip()
    if not text or text.lower() == "all":
        return [area.code for area in LOAD_AREAS]
    codes: list[str] = []
    seen: set[str] = set()
    known = ", ".join(area.code for area in LOAD_AREAS)
    for part in text.split(","):
        code = part.strip().upper()
        if not code or code in seen:
            continue
        if code not in BY_CODE:
            raise ValueError(f"Unknown load area {code!r}. Known load areas: {known}")
        seen.add(code)
        codes.append(code)
    if not codes:
        raise ValueError("No load areas selected")
    return codes
