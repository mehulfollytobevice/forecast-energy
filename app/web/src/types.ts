export type Level = "pjm" | "mkt_region" | "zone" | "load_area";

export interface ForecastNode {
  id: string;
  level: Level;
  parent_id: string | null;
  name: string;
  timestamps: string[];
  p10: number[];
  p50: number[];
  p90: number[];
  actual_timestamps: string[];
  actuals: number[];
  /** PJM's own forecast aligned to `timestamps`; null where PJM has no value. */
  benchmark?: (number | null)[];
}

export interface BenchmarkInfo {
  source: string;
  feed: string;
  issued_at: string;
}

export interface ForecastSnapshot {
  generated_at: string;
  origin: string;
  timezone: string;
  nodes: ForecastNode[];
  model_version?: string;
  weather_run?: string;
  load_through?: string;
  model_sources?: Record<string, string>;
  benchmark?: BenchmarkInfo;
}

export interface BacktestMetrics {
  n: number;
  mae: number;
  rmse: number;
  pinball_p10?: number;
  pinball_p50?: number;
  pinball_p90?: number;
  coverage_p10_p90?: number;
}

export interface BacktestNode {
  id: string;
  level: Level;
  parent_id: string | null;
  name: string;
  timestamps: string[];
  actuals: number[];
  p10: number[];
  p50: number[];
  p90: number[];
  metrics?: BacktestMetrics;
}

export interface BacktestSnapshot {
  generated_at: string;
  timezone: string;
  start: string;
  end: string;
  lead_hours: number;
  nodes: BacktestNode[];
}

export interface GeoProperties {
  id: string;
  parent_id: string | null;
  level: Level;
  name: string;
  approximate?: boolean;
}

export interface Focus {
  id: string;
  level: Level;
  name: string;
}

export type DashboardTab = "live" | "backtest";

/** Nodes the map/hierarchy helpers can index. */
export type HierarchyNode = ForecastNode | BacktestNode;
