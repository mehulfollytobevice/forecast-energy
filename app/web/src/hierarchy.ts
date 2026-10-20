import type { Feature, FeatureCollection, GeoJsonProperties, Geometry } from "geojson";

import type { BacktestNode, Focus, GeoProperties, HierarchyNode, Level } from "./types";
import { nodeKey } from "./format";

const MULTI_AREA_ZONES = new Set(["AE", "AEP", "ATSI", "DPL", "PEP", "PL"]);

export function buildNodeIndex<T extends HierarchyNode>(nodes: T[]): Map<string, T> {
  const index = new Map<string, T>();
  for (const node of nodes) {
    index.set(nodeKey(node.id, node.level), node);
  }
  return index;
}

export function defaultPath(): Focus[] {
  return [{ id: "PJM", level: "pjm", name: "PJM" }];
}

export function mapLevel(focus: Focus): Level {
  if (focus.level === "pjm") return "mkt_region";
  if (focus.level === "mkt_region") return "zone";
  if (focus.level === "zone") {
    return MULTI_AREA_ZONES.has(focus.id) ? "load_area" : "zone";
  }
  return "load_area";
}

export function filterMapFeatures(
  collection: FeatureCollection,
  focus: Focus,
): Feature<Geometry, GeoProperties>[] {
  const level = mapLevel(focus);
  return collection.features.filter((feature) => {
    const props = feature.properties as GeoProperties | null;
    if (!props || props.level !== level) return false;
    if (focus.level === "pjm") return props.parent_id === "PJM";
    if (focus.level === "mkt_region") return props.parent_id === focus.id;
    if (focus.level === "zone" && MULTI_AREA_ZONES.has(focus.id)) {
      return props.parent_id === focus.id;
    }
    if (focus.level === "zone") return props.id === focus.id;
    return props.id === focus.id;
  }) as Feature<Geometry, GeoProperties>[];
}

export function cardsForFocus<T extends HierarchyNode>(
  collection: FeatureCollection,
  focus: Focus,
  index: Map<string, T>,
): T[] {
  if (focus.level === "load_area") {
    const leaf = index.get(nodeKey(focus.id, focus.level));
    return leaf ? [leaf] : [];
  }
  const features = filterMapFeatures(collection, focus);
  const nodes: T[] = [];
  for (const feature of features) {
    const props = feature.properties as GeoProperties;
    const node = index.get(nodeKey(props.id, props.level));
    if (node) nodes.push(node);
  }
  nodes.sort((a, b) => a.name.localeCompare(b.name));
  return nodes;
}

export function drillTarget(
  focus: Focus,
  props: GeoProperties,
): Focus | null {
  if (focus.level === "load_area") return null;
  if (focus.level === "zone" && !MULTI_AREA_ZONES.has(focus.id)) {
    if (props.level === "zone" && props.id === focus.id) {
      return { id: props.id, level: "load_area", name: props.name };
    }
    return null;
  }
  if (props.level === "mkt_region") {
    return { id: props.id, level: "mkt_region", name: props.name };
  }
  if (props.level === "zone") {
    return { id: props.id, level: "zone", name: props.name };
  }
  if (props.level === "load_area") {
    return { id: props.id, level: "load_area", name: props.name };
  }
  return null;
}

export function pathToFocus(path: Focus[]): Focus {
  return path[path.length - 1];
}

export function truncatePath(path: Focus[], index: number): Focus[] {
  return path.slice(0, index + 1);
}

export type HoverInfo = {
  name: string;
  kind: "forecast" | "backtest";
  p50?: number;
  p10?: number;
  p90?: number;
  mae?: number;
  rmse?: number;
  coverage?: number;
};

export function hoverForFeature(
  props: GeoProperties,
  index: Map<string, HierarchyNode>,
  colorBy: "p50" | "mae" = "p50",
): HoverInfo | null {
  const node = index.get(nodeKey(props.id, props.level));
  if (!node) return null;
  if (colorBy === "mae") {
    const metrics = "metrics" in node ? (node as BacktestNode).metrics : undefined;
    if (!metrics) return { name: node.name, kind: "backtest" };
    return {
      name: node.name,
      kind: "backtest",
      mae: metrics.mae,
      rmse: metrics.rmse,
      coverage: metrics.coverage_p10_p90,
    };
  }
  if (node.p50.length === 0) return null;
  return {
    name: node.name,
    kind: "forecast",
    p50: node.p50[0],
    p10: node.p10[0],
    p90: node.p90[0],
  };
}

export function colorValueForNode(
  node: HierarchyNode | undefined,
  colorBy: "p50" | "mae",
): number | undefined {
  if (!node) return undefined;
  if (colorBy === "mae") {
    const metrics = "metrics" in node ? (node as BacktestNode).metrics : undefined;
    return metrics?.mae;
  }
  return node.p50[0];
}

export function emptyCollection(): FeatureCollection {
  return { type: "FeatureCollection", features: [] };
}

export function isGeoProperties(
  value: GeoJsonProperties | undefined | null,
): value is GeoProperties {
  if (!value || typeof value !== "object") return false;
  return "id" in value && "level" in value && "name" in value;
}
