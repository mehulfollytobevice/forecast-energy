import type { BacktestMetrics, BacktestNode } from "./types";

export const EASTERN_TZ = "America/New_York";

/** Inclusive calendar-day window in US/Eastern (`YYYY-MM-DD`). */
export interface DateWindow {
  start: string;
  end: string;
}

/** Convert an ISO timestamp to an Eastern calendar date key. */
export function toEasternDateKey(iso: string): string {
  return new Date(iso).toLocaleDateString("en-CA", { timeZone: EASTERN_TZ });
}

/** Earliest and latest Eastern calendar days present in the series. */
export function nodeDateRange(node: BacktestNode): DateWindow | null {
  if (node.timestamps.length === 0) return null;
  return {
    start: toEasternDateKey(node.timestamps[0]),
    end: toEasternDateKey(node.timestamps[node.timestamps.length - 1]),
  };
}

/** Union of calendar ranges across nodes that have data. */
export function unionDateRange(nodes: BacktestNode[]): DateWindow | null {
  const ranges = nodes
    .map(nodeDateRange)
    .filter((range): range is DateWindow => range !== null);
  if (ranges.length === 0) return null;
  let start = ranges[0].start;
  let end = ranges[0].end;
  for (let i = 1; i < ranges.length; i++) {
    if (ranges[i].start < start) start = ranges[i].start;
    if (ranges[i].end > end) end = ranges[i].end;
  }
  return { start, end };
}

/** Keep `window` inside `bounds`, or reset when the range collapses. */
export function clampDateWindow(window: DateWindow, bounds: DateWindow): DateWindow {
  let start = window.start < bounds.start ? bounds.start : window.start;
  let end = window.end > bounds.end ? bounds.end : window.end;
  if (start > bounds.end) start = bounds.start;
  if (end < bounds.start) end = bounds.end;
  if (start > end) return { start: bounds.start, end: bounds.end };
  return { start, end };
}

export function formatDateKey(dateKey: string): string {
  const [year, month, day] = dateKey.split("-").map(Number);
  const utc = new Date(Date.UTC(year, month - 1, day, 12));
  return utc.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

function pinball(actual: number[], forecast: number[], quantile: number): number {
  let total = 0;
  for (let i = 0; i < actual.length; i++) {
    const err = actual[i] - forecast[i];
    total += err >= 0 ? quantile * err : (quantile - 1) * err;
  }
  return total / actual.length;
}

/** Score the sliced series the same way as the Python snapshot builder. */
export function scoreBacktestSlice(
  actuals: number[],
  p50: number[],
  p10: number[],
  p90: number[],
): BacktestMetrics | undefined {
  if (actuals.length === 0 || actuals.length !== p50.length) return undefined;
  const residual = actuals.map((a, i) => a - p50[i]);
  const mae = residual.reduce((sum, r) => sum + Math.abs(r), 0) / residual.length;
  const rmse = Math.sqrt(residual.reduce((sum, r) => sum + r * r, 0) / residual.length);
  const metrics: BacktestMetrics = {
    n: actuals.length,
    mae,
    rmse,
  };
  if (p10.length === actuals.length) {
    metrics.pinball_p10 = pinball(actuals, p10, 0.1);
  }
  metrics.pinball_p50 = pinball(actuals, p50, 0.5);
  if (p90.length === actuals.length) {
    metrics.pinball_p90 = pinball(actuals, p90, 0.9);
  }
  if (p10.length === actuals.length && p90.length === actuals.length) {
    let inside = 0;
    for (let i = 0; i < actuals.length; i++) {
      if (actuals[i] >= p10[i] && actuals[i] <= p90[i]) inside += 1;
    }
    metrics.coverage_p10_p90 = inside / actuals.length;
  }
  return metrics;
}

/** Slice a node to Eastern calendar days `[start, end]` and recompute metrics. */
export function sliceBacktestNode(
  node: BacktestNode,
  window: DateWindow,
): BacktestNode {
  const indexes: number[] = [];
  for (let i = 0; i < node.timestamps.length; i++) {
    const key = toEasternDateKey(node.timestamps[i]);
    if (key >= window.start && key <= window.end) indexes.push(i);
  }
  const timestamps = indexes.map((i) => node.timestamps[i]);
  const actuals = indexes.map((i) => node.actuals[i]);
  const p10 = indexes.map((i) => node.p10[i]);
  const p50 = indexes.map((i) => node.p50[i]);
  const p90 = indexes.map((i) => node.p90[i]);
  const metrics = scoreBacktestSlice(actuals, p50, p10, p90);
  const { metrics: _ignored, ...rest } = node;
  return {
    ...rest,
    timestamps,
    actuals,
    p10,
    p50,
    p90,
    ...(metrics ? { metrics } : {}),
  };
}

export function sliceBacktestNodes(
  nodes: BacktestNode[],
  window: DateWindow,
): BacktestNode[] {
  return nodes.map((node) => sliceBacktestNode(node, window));
}
