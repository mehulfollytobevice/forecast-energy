import type { ForecastNode } from "./types";
import { formatMw } from "./format";

const WIDTH = 960;
const HEIGHT = 280;
const MARGIN = { top: 16, right: 20, bottom: 44, left: 64 };
const PLOT_W = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_H = HEIGHT - MARGIN.top - MARGIN.bottom;

interface FanChartProps {
  node: ForecastNode;
  onSelect?: (node: ForecastNode) => void;
}

export function FanChart({ node, onSelect }: FanChartProps) {
  const history = node.actuals;
  const forecast = node.p50;
  const p10 = node.p10;
  const p90 = node.p90;
  const times = [...node.actual_timestamps, ...node.timestamps];
  const series = [...history, ...forecast];
  const bandLow = [...history, ...p10];
  const bandHigh = [...history, ...p90];
  const benchmark = node.benchmark?.length === forecast.length ? node.benchmark : null;
  const benchmarkValues = (benchmark ?? []).filter((v): v is number => v !== null);

  if (series.length < 2 || times.length !== series.length) {
    return <div className="fan empty">No series</div>;
  }

  const min = Math.min(...bandLow, ...bandHigh, ...benchmarkValues);
  const max = Math.max(...bandLow, ...bandHigh, ...benchmarkValues);
  const pad = (max - min) * 0.08 || 1;
  const yMin = Math.max(0, min - pad);
  const yMax = max + pad;
  const span = yMax - yMin || 1;

  const x = (i: number) => MARGIN.left + (i / (series.length - 1)) * PLOT_W;
  const y = (v: number) => MARGIN.top + PLOT_H - ((v - yMin) / span) * PLOT_H;

  const linePath = (values: number[], start: number) =>
    values
      .map((v, i) => `${i === 0 ? "M" : "L"} ${x(start + i).toFixed(1)} ${y(v).toFixed(1)}`)
      .join(" ");

  const split = history.length;
  const bandPath =
    linePath(bandLow.slice(Math.max(0, split - 1)), Math.max(0, split - 1)) +
    " " +
    [...bandHigh.slice(Math.max(0, split - 1))]
      .reverse()
      .map((v, i) => {
        const idx = bandHigh.length - 1 - i;
        return `L ${x(idx).toFixed(1)} ${y(v).toFixed(1)}`;
      })
      .join(" ") +
    " Z";

  // Break the line wherever PJM has no value for an hour.
  const benchmarkPath = (benchmark ?? [])
    .map((v, i) => {
      if (v === null) return "";
      const cmd = i > 0 && benchmark?.[i - 1] !== null ? "L" : "M";
      return `${cmd} ${x(split + i).toFixed(1)} ${y(v).toFixed(1)}`;
    })
    .filter(Boolean)
    .join(" ");

  const yTicks = niceTicks(yMin, yMax, 5);
  const xTickIndexes = timeTickIndexes(times.length);
  const nextP50 = forecast[0] ?? series[series.length - 1];
  const nextBenchmark = benchmark?.[0] ?? null;
  const interactive = Boolean(onSelect);

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
            {history.length}h history · {forecast.length}h forecast · P10–P90 band
          </p>
        </div>
        <div className="fan-stats">
          <div className="fan-stat">
            <span className="fan-stat-label">Next hour P50</span>
            <span className="fan-stat-value">{formatMw(nextP50)}</span>
          </div>
          {nextBenchmark !== null && (
            <div className="fan-stat fan-stat-benchmark">
              <span className="fan-stat-label">PJM forecast</span>
              <span className="fan-stat-value">{formatMw(nextBenchmark)}</span>
            </div>
          )}
        </div>
      </div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`${node.name} load forecast`}
        className="fan-svg"
      >
        {/* Plot frame */}
        <rect
          x={MARGIN.left}
          y={MARGIN.top}
          width={PLOT_W}
          height={PLOT_H}
          fill="#f8fafc"
          stroke="#e2e8f0"
        />

        {/* Y grid + labels */}
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
              x={MARGIN.left - 10}
              y={y(tick)}
              textAnchor="end"
              dominantBaseline="middle"
              className="fan-axis-label"
            >
              {formatAxisMw(tick)}
            </text>
          </g>
        ))}

        {/* X ticks + labels */}
        {xTickIndexes.map((i) => (
          <g key={`x-${i}`}>
            <line
              x1={x(i)}
              x2={x(i)}
              y1={MARGIN.top + PLOT_H}
              y2={MARGIN.top + PLOT_H + 6}
              stroke="#94a3b8"
            />
            <text
              x={x(i)}
              y={MARGIN.top + PLOT_H + 22}
              textAnchor="middle"
              className="fan-axis-label"
            >
              {formatTickTime(times[i])}
            </text>
            <text
              x={x(i)}
              y={MARGIN.top + PLOT_H + 36}
              textAnchor="middle"
              className="fan-axis-label fan-axis-date"
            >
              {formatTickDate(times[i])}
            </text>
          </g>
        ))}

        {/* Axis titles */}
        <text
          x={14}
          y={MARGIN.top + PLOT_H / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${MARGIN.top + PLOT_H / 2})`}
          className="fan-axis-title"
        >
          Load (MW)
        </text>
        <text
          x={MARGIN.left + PLOT_W / 2}
          y={HEIGHT - 2}
          textAnchor="middle"
          className="fan-axis-title"
        >
          Time (US/Eastern)
        </text>

        {/* Forecast uncertainty band */}
        {forecast.length > 0 && <path d={bandPath} fill="rgba(37, 99, 235, 0.2)" stroke="none" />}

        {/* History */}
        {history.length > 1 && (
          <path d={linePath(history, 0)} fill="none" stroke="#0f172a" strokeWidth="2" />
        )}

        {/* Forecast median */}
        {forecast.length > 1 && (
          <path
            d={linePath(
              split > 0 ? [history[history.length - 1], ...forecast] : forecast,
              Math.max(0, split - 1),
            )}
            fill="none"
            stroke="#1d4ed8"
            strokeWidth="2.25"
          />
        )}

        {/* PJM benchmark */}
        {benchmarkPath && (
          <path
            d={benchmarkPath}
            fill="none"
            stroke="#ea580c"
            strokeWidth="2"
            strokeDasharray="6 4"
          />
        )}

        {/* Now marker */}
        {split > 0 && split < series.length && (
          <g>
            <line
              x1={x(split - 0.5)}
              x2={x(split - 0.5)}
              y1={MARGIN.top}
              y2={MARGIN.top + PLOT_H}
              stroke="#64748b"
              strokeDasharray="4 4"
              strokeWidth="1.5"
            />
            <text
              x={x(split - 0.5) + 6}
              y={MARGIN.top + 14}
              className="fan-now-label"
            >
              forecast →
            </text>
          </g>
        )}
      </svg>
      <div className="fan-legend" aria-hidden="true">
        <span>
          <i className="swatch actual" /> Actual
        </span>
        <span>
          <i className="swatch p50" /> P50 forecast
        </span>
        <span>
          <i className="swatch band" /> P10–P90
        </span>
        {benchmarkPath && (
          <span>
            <i className="swatch benchmark" /> PJM forecast
          </span>
        )}
      </div>
    </article>
  );
}

function formatAxisMw(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 10_000) return `${(value / 1000).toFixed(1)} GW`;
  if (abs >= 1000) return `${Math.round(value).toLocaleString()}`;
  if (abs >= 10) return `${Math.round(value)}`;
  return value.toFixed(1);
}

function formatTickTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
    timeZone: "America/New_York",
  });
}

function formatTickDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "America/New_York",
  });
}

/** Pick ~5–7 evenly spaced indexes, always including first and last. */
function timeTickIndexes(n: number): number[] {
  if (n <= 2) return Array.from({ length: n }, (_, i) => i);
  const target = Math.min(7, Math.max(4, Math.round(n / 12)));
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
