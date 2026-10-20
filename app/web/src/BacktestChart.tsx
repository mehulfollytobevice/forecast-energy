import { formatDateKey } from "./backtestWindow";
import { formatMw } from "./format";
import type { BacktestNode } from "./types";

const WIDTH = 960;
const HEIGHT = 280;
const MARGIN = { top: 16, right: 20, bottom: 44, left: 64 };
const PLOT_W = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_H = HEIGHT - MARGIN.top - MARGIN.bottom;
const MAX_POINTS = 800;

interface BacktestChartProps {
  node: BacktestNode;
  /** Selected Eastern calendar day (`YYYY-MM-DD`). */
  windowStart: string;
  windowEnd: string;
  /** Region's own Eastern data span (`YYYY-MM-DD`), when known. */
  availableStart?: string;
  availableEnd?: string;
  leadHours: number;
  onSelect?: (node: BacktestNode) => void;
}

export function BacktestChart({
  node,
  windowStart,
  windowEnd,
  availableStart,
  availableEnd,
  leadHours,
  onSelect,
}: BacktestChartProps) {
  const n = node.timestamps.length;
  if (n < 2 || node.actuals.length !== n || node.p50.length !== n) {
    const availableLabel =
      availableStart && availableEnd
        ? ` · data ${formatDateKey(availableStart)} – ${formatDateKey(availableEnd)}`
        : "";
    return (
      <article className="fan">
        <div className="fan-head">
          <div>
            <h2>{node.name}</h2>
            <p className="fan-subtitle">
              No overlap with {formatDateKey(windowStart)} – {formatDateKey(windowEnd)}
              {availableLabel}
            </p>
          </div>
        </div>
        <div className="fan empty">No backtest series in the selected window</div>
      </article>
    );
  }

  const stride = Math.max(1, Math.ceil(n / MAX_POINTS));
  const indexes = strideIndexes(n, stride);
  const times = indexes.map((i) => node.timestamps[i]);
  const actuals = indexes.map((i) => node.actuals[i]);
  const p50 = indexes.map((i) => node.p50[i]);
  const p10 = node.p10.length === n ? indexes.map((i) => node.p10[i]) : null;
  const p90 = node.p90.length === n ? indexes.map((i) => node.p90[i]) : null;

  const bandLow = p10 ?? actuals;
  const bandHigh = p90 ?? actuals;
  const min = Math.min(...bandLow, ...bandHigh, ...actuals, ...p50);
  const max = Math.max(...bandLow, ...bandHigh, ...actuals, ...p50);
  const pad = (max - min) * 0.08 || 1;
  const yMin = Math.max(0, min - pad);
  const yMax = max + pad;
  const span = yMax - yMin || 1;

  const m = times.length;
  const x = (i: number) => MARGIN.left + (i / Math.max(1, m - 1)) * PLOT_W;
  const y = (v: number) => MARGIN.top + PLOT_H - ((v - yMin) / span) * PLOT_H;

  const linePath = (values: number[]) =>
    values
      .map((v, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`)
      .join(" ");

  const bandPath =
    p10 && p90
      ? linePath(p10) +
        " " +
        [...p90]
          .reverse()
          .map((v, i) => `L ${x(m - 1 - i).toFixed(1)} ${y(v).toFixed(1)}`)
          .join(" ") +
        " Z"
      : null;

  const yTicks = niceTicks(yMin, yMax, 5);
  const xTickIndexes = timeTickIndexes(m);
  const interactive = Boolean(onSelect);
  const metrics = node.metrics;

  return (
    <article className={`fan${interactive ? " fan-interactive" : ""}`}>
      <div className="fan-head">
        <div>
          <h2>
            {interactive ? (
              <button type="button" className="fan-title-btn" onClick={() => onSelect?.(node)}>
                {node.name}
              </button>
            ) : (
              node.name
            )}
          </h2>
          <p className="fan-subtitle">
            {formatDateKey(windowStart)} – {formatDateKey(windowEnd)} · {leadHours}h-ahead ·
            actual vs P50
            {availableStart &&
            availableEnd &&
            (availableStart !== windowStart || availableEnd !== windowEnd)
              ? ` · region data ${formatDateKey(availableStart)} – ${formatDateKey(availableEnd)}`
              : ""}
            {stride > 1 ? ` · display stride ${stride}` : ""}
          </p>
        </div>
        {metrics && (
          <div className="metrics-row" aria-label={`${node.name} backtest metrics`}>
            <div className="fan-stat">
              <span className="fan-stat-label">MAE</span>
              <span className="fan-stat-value">{formatMw(metrics.mae)}</span>
            </div>
            <div className="fan-stat">
              <span className="fan-stat-label">RMSE</span>
              <span className="fan-stat-value">{formatMw(metrics.rmse)}</span>
            </div>
            {metrics.coverage_p10_p90 != null && (
              <div className="fan-stat">
                <span className="fan-stat-label">P10–P90</span>
                <span className="fan-stat-value">
                  {(metrics.coverage_p10_p90 * 100).toFixed(0)}%
                </span>
              </div>
            )}
          </div>
        )}
      </div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`${node.name} backtest actual versus forecast`}
        className="fan-svg"
      >
        <rect
          x={MARGIN.left}
          y={MARGIN.top}
          width={PLOT_W}
          height={PLOT_H}
          fill="#f8fafc"
          stroke="#e2e8f0"
        />
        {yTicks.map((tick) => (
          <g key={`y-${tick}`}>
            <line
              x1={MARGIN.left}
              x2={MARGIN.left + PLOT_W}
              y1={y(tick)}
              y2={y(tick)}
              stroke="#e2e8f0"
              strokeWidth="1"
            />
            <text
              x={MARGIN.left - 8}
              y={y(tick) + 4}
              textAnchor="end"
              className="fan-axis-label"
            >
              {formatAxisMw(tick)}
            </text>
          </g>
        ))}
        {xTickIndexes.map((i) => (
          <g key={`x-${i}`}>
            <line
              x1={x(i)}
              x2={x(i)}
              y1={MARGIN.top + PLOT_H}
              y2={MARGIN.top + PLOT_H + 4}
              stroke="#94a3b8"
            />
            <text x={x(i)} y={MARGIN.top + PLOT_H + 18} textAnchor="middle" className="fan-axis-date">
              {formatTickDate(times[i])}
            </text>
          </g>
        ))}
        <text
          x={12}
          y={MARGIN.top + PLOT_H / 2}
          transform={`rotate(-90 12 ${MARGIN.top + PLOT_H / 2})`}
          textAnchor="middle"
          className="fan-axis-title"
        >
          Load (MW)
        </text>
        <text
          x={MARGIN.left + PLOT_W / 2}
          y={HEIGHT - 6}
          textAnchor="middle"
          className="fan-axis-title"
        >
          Time (US/Eastern)
        </text>
        {bandPath && <path d={bandPath} fill="rgba(37, 99, 235, 0.2)" stroke="none" />}
        <path d={linePath(actuals)} fill="none" stroke="#0f172a" strokeWidth="2" />
        <path d={linePath(p50)} fill="none" stroke="#1d4ed8" strokeWidth="2" />
      </svg>
      <div className="fan-legend" aria-hidden="true">
        <span>
          <i className="swatch actual" /> Actual
        </span>
        <span>
          <i className="swatch p50" /> P50 ({leadHours}h-ahead)
        </span>
        {bandPath && (
          <span>
            <i className="swatch band" /> P10–P90
          </span>
        )}
      </div>
    </article>
  );
}

function strideIndexes(n: number, stride: number): number[] {
  if (stride <= 1) return Array.from({ length: n }, (_, i) => i);
  const indexes: number[] = [];
  for (let i = 0; i < n; i += stride) indexes.push(i);
  if (indexes[indexes.length - 1] !== n - 1) indexes.push(n - 1);
  return indexes;
}

function formatAxisMw(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 10_000) return `${(value / 1000).toFixed(1)} GW`;
  if (abs >= 1000) return `${Math.round(value).toLocaleString()}`;
  if (abs >= 10) return `${Math.round(value)}`;
  return value.toFixed(1);
}

function formatTickDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "America/New_York",
  });
}

function timeTickIndexes(n: number): number[] {
  if (n <= 2) return Array.from({ length: n }, (_, i) => i);
  const target = Math.min(7, Math.max(4, Math.round(n / 24)));
  const indexes = new Set<number>();
  for (let t = 0; t < target; t++) {
    indexes.add(Math.round((t / (target - 1)) * (n - 1)));
  }
  return [...indexes].sort((a, b) => a - b);
}

function niceTicks(min: number, max: number, count: number): number[] {
  if (!(max > min)) return [min];
  const span = max - min;
  const step = niceNumber(span / Math.max(1, count - 1), true);
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 1e-6; v += step) {
    ticks.push(Number(v.toPrecision(10)));
  }
  return ticks.length ? ticks : [min, max];
}

function niceNumber(range: number, round: boolean): number {
  const exp = Math.floor(Math.log10(range));
  const frac = range / 10 ** exp;
  let nice: number;
  if (round) {
    if (frac < 1.5) nice = 1;
    else if (frac < 3) nice = 2;
    else if (frac < 7) nice = 5;
    else nice = 10;
  } else if (frac <= 1) nice = 1;
  else if (frac <= 2) nice = 2;
  else if (frac <= 5) nice = 5;
  else nice = 10;
  return nice * 10 ** exp;
}
