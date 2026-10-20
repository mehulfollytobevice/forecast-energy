import type { FeatureCollection } from "geojson";
import maplibregl from "maplibre-gl";
import { useEffect, useRef } from "react";

import { formatMw, lerpColor, nodeKey } from "./format";
import {
  buildNodeIndex,
  colorValueForNode,
  filterMapFeatures,
  hoverForFeature,
  isGeoProperties,
} from "./hierarchy";
import type { Focus, GeoProperties, HierarchyNode } from "./types";

const SOURCE = "pjm-children";
const FILL = "pjm-fill";
const LINE = "pjm-line";
const COLORS = ["#eff6ff", "#2563eb", "#1e3a8a"];
const MAE_COLORS = ["#fef3c7", "#f59e0b", "#b45309"];

/** Camera clamp around PJM — loose enough to leave margin around the footprint. */
const PJM_BOUNDS: maplibregl.LngLatBoundsLike = [
  [-94.0, 32.5],
  [-69.0, 45.5],
];
const PJM_MIN_ZOOM = 4.2;
const PJM_MAX_ZOOM = 9;
const FIT_PADDING = { top: 72, bottom: 72, left: 88, right: 88 };

interface ForecastMapProps {
  geo: FeatureCollection;
  focus: Focus;
  nodes: HierarchyNode[];
  onDrill: (props: GeoProperties) => void;
  colorBy?: "p50" | "mae";
}

export function ForecastMap({
  geo,
  focus,
  nodes,
  onDrill,
  colorBy = "p50",
}: ForecastMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const onDrillRef = useRef(onDrill);
  const colorByRef = useRef(colorBy);
  const indexRef = useRef(buildNodeIndex(nodes));

  useEffect(() => {
    onDrillRef.current = onDrill;
  }, [onDrill]);

  useEffect(() => {
    colorByRef.current = colorBy;
  }, [colorBy]);

  useEffect(() => {
    indexRef.current = buildNodeIndex(nodes);
  }, [nodes]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap",
            bounds: [-94.0, 32.5, -69.0, 45.5],
          },
        },
        layers: [
          {
            id: "background",
            type: "background",
            paint: { "background-color": "#e8eef5" },
          },
          { id: "osm", type: "raster", source: "osm" },
        ],
      },
      center: [-78.5, 40.2],
      zoom: PJM_MIN_ZOOM,
      minZoom: PJM_MIN_ZOOM,
      maxZoom: PJM_MAX_ZOOM,
      maxBounds: PJM_BOUNDS,
      renderWorldCopies: false,
      dragRotate: false,
      pitchWithRotate: false,
    });
    map.touchZoomRotate.disableRotation();
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.on("load", () => {
      map.addSource(SOURCE, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: FILL,
        type: "fill",
        source: SOURCE,
        paint: { "fill-color": ["get", "fillColor"], "fill-opacity": 0.82 },
      });
      map.addLayer({
        id: LINE,
        type: "line",
        source: SOURCE,
        paint: { "line-color": "#334155", "line-width": 1 },
      });
      map.on("mousemove", FILL, (event) => {
        const props = event.features?.[0]?.properties;
        if (!isGeoProperties(props)) return;
        const info = hoverForFeature(props, indexRef.current, colorByRef.current);
        if (!info) return;
        map.getCanvas().style.cursor = "pointer";
        const body =
          info.kind === "backtest"
            ? [
                info.mae != null ? `MAE ${formatMw(info.mae)}` : null,
                info.rmse != null ? `RMSE ${formatMw(info.rmse)}` : null,
                info.coverage != null
                  ? `Coverage ${(info.coverage * 100).toFixed(0)}%`
                  : null,
              ]
                .filter(Boolean)
                .join("<br/>")
            : `P50 ${formatMw(info.p50!)}<br/>P10–P90 ${formatMw(info.p10!)} – ${formatMw(info.p90!)}`;
        const html = `<div class="map-popup"><strong>${info.name}</strong><br/>${body}</div>`;
        if (!popupRef.current) {
          popupRef.current = new maplibregl.Popup({
            closeButton: false,
            closeOnClick: false,
            offset: 8,
          });
        }
        popupRef.current.setLngLat(event.lngLat).setHTML(html).addTo(map);
      });
      map.on("mouseleave", FILL, () => {
        map.getCanvas().style.cursor = "";
        popupRef.current?.remove();
      });
      map.on("click", FILL, (event) => {
        const props = event.features?.[0]?.properties;
        if (isGeoProperties(props)) onDrillRef.current(props);
      });
    });
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const palette = colorBy === "mae" ? MAE_COLORS : COLORS;
    const apply = () => {
      const visible = filterMapFeatures(geo, focus);
      const values = visible
        .map((feature) => {
          const props = feature.properties as GeoProperties;
          return colorValueForNode(
            indexRef.current.get(nodeKey(props.id, props.level)),
            colorBy,
          );
        })
        .filter((v): v is number => v != null);
      const min = values.length ? Math.min(...values) : 0;
      const max = values.length ? Math.max(...values) : 1;
      const span = max - min || 1;
      const enriched: FeatureCollection = {
        type: "FeatureCollection",
        features: visible.map((feature) => {
          const props = feature.properties as GeoProperties;
          const node = indexRef.current.get(nodeKey(props.id, props.level));
          const value = colorValueForNode(node, colorBy) ?? min;
          const t = (value - min) / span;
          return {
            ...feature,
            properties: {
              ...props,
              colorValue: value,
              fillColor: lerpColor(t, palette[0], palette[2]),
            },
          };
        }),
      };
      const source = map.getSource(SOURCE) as maplibregl.GeoJSONSource | undefined;
      source?.setData(enriched);
      const bounds = new maplibregl.LngLatBounds();
      for (const feature of visible) {
        walkCoords(feature.geometry, (lng, lat) => bounds.extend([lng, lat]));
      }
      if (!bounds.isEmpty()) {
        map.fitBounds(bounds, {
          padding: FIT_PADDING,
          duration: 600,
          maxZoom: PJM_MAX_ZOOM,
        });
      }
    };
    if (map.isStyleLoaded()) apply();
    else map.once("load", apply);
  }, [geo, focus, nodes, colorBy]);

  return <div className="map-wrap" ref={containerRef} />;
}

function walkCoords(geometry: GeoJSON.Geometry, visit: (lng: number, lat: number) => void) {
  if (geometry.type === "Polygon") {
    for (const ring of geometry.coordinates) {
      for (const [lng, lat] of ring) visit(lng, lat);
    }
  } else if (geometry.type === "MultiPolygon") {
    for (const poly of geometry.coordinates) {
      for (const ring of poly) {
        for (const [lng, lat] of ring) visit(lng, lat);
      }
    }
  }
}
