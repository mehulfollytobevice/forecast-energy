export interface TooltipRow {
  label: string;
  value: string;
  swatch?: "actual" | "p50" | "band" | "benchmark";
}

interface ChartTooltipProps {
  leftPct: number;
  align: "left" | "right";
  timestamp: string;
  rows: TooltipRow[];
}

export function ChartTooltip({ leftPct, align, timestamp, rows }: ChartTooltipProps) {
  return (
    <div
      className={`fan-tooltip fan-tooltip-${align}`}
      style={{ left: `${leftPct}%` }}
      role="status"
    >
      <div className="fan-tooltip-time">{timestamp}</div>
      {rows.map((row) => (
        <div key={row.label} className="fan-tooltip-row">
          <span className="fan-tooltip-label">
            {row.swatch ? <i className={`swatch ${row.swatch}`} /> : null}
            {row.label}
          </span>
          <span className="fan-tooltip-value">{row.value}</span>
        </div>
      ))}
    </div>
  );
}
