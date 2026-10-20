import type { DateWindow } from "./backtestWindow";
import { formatDateKey } from "./backtestWindow";

interface BacktestDateFilterProps {
  available: DateWindow;
  value: DateWindow;
  onChange: (next: DateWindow) => void;
}

export function BacktestDateFilter({ available, value, onChange }: BacktestDateFilterProps) {
  return (
    <div className="date-filter" aria-label="Backtest date range">
      <div className="date-filter-fields">
        <label className="date-field">
          <span className="meta-label">From</span>
          <input
            type="date"
            value={value.start}
            min={available.start}
            max={value.end < available.end ? value.end : available.end}
            onChange={(event) => {
              const start = event.target.value;
              if (!start) return;
              onChange({
                start,
                end: start > value.end ? start : value.end,
              });
            }}
          />
        </label>
        <label className="date-field">
          <span className="meta-label">To</span>
          <input
            type="date"
            value={value.end}
            min={value.start > available.start ? value.start : available.start}
            max={available.end}
            onChange={(event) => {
              const end = event.target.value;
              if (!end) return;
              onChange({
                start: end < value.start ? end : value.start,
                end,
              });
            }}
          />
        </label>
        {(value.start !== available.start || value.end !== available.end) && (
          <button
            type="button"
            className="date-reset"
            onClick={() => onChange({ ...available })}
          >
            Full range
          </button>
        )}
      </div>
      <p className="date-filter-hint">
        Data available in this view: {formatDateKey(available.start)} –{" "}
        {formatDateKey(available.end)}
      </p>
    </div>
  );
}
