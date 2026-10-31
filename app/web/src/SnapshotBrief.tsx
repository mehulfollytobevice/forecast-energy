import { formatEasternParts, formatEasternShort, formatRelativeTime, formatWeatherRun } from "./format";
import type { BacktestSnapshot, DashboardTab, ForecastSnapshot } from "./types";

interface SnapshotBriefProps {
  tab: DashboardTab;
  live: ForecastSnapshot;
  backtest: BacktestSnapshot | null;
  stale: boolean;
}

interface Fact {
  label: string;
  value: string;
  detail?: string;
  hint: string;
}

export function SnapshotBrief({ tab, live, backtest, stale }: SnapshotBriefProps) {
  if (tab === "backtest") {
    return <BacktestBrief backtest={backtest} />;
  }
  return <LiveBrief live={live} stale={stale} />;
}

function LiveBrief({ live, stale }: { live: ForecastSnapshot; stale: boolean }) {
  const origin = formatEasternParts(live.origin);
  const load = live.load_through ? formatEasternParts(live.load_through) : null;
  const weather = live.weather_run ? formatWeatherRun(live.weather_run) : null;
  const pjm = live.benchmark ? formatEasternParts(live.benchmark.issued_at) : null;
  const published = formatRelativeTime(live.generated_at);

  const facts: Fact[] = [
    {
      label: "Forecast starts",
      value: origin.time,
      detail: origin.date,
      hint: "First hour of the 48-hour outlook",
    },
  ];
  if (load) {
    facts.push({
      label: "Last observed load",
      value: load.time,
      detail: load.date,
      hint: "Newest demand the model could use",
    });
  }
  if (weather) {
    facts.push({
      label: "Weather used",
      value: weather.time || weather.date,
      detail: weather.time ? weather.date : undefined,
      hint: "ECMWF cycle — no later updates",
    });
  }
  if (pjm) {
    facts.push({
      label: "PJM official",
      value: pjm.time,
      detail: pjm.date,
      hint: "When the dashed orange line was issued",
    });
  }

  return (
    <section className="brief" aria-label="How to read this forecast">
      <div className="brief-top">
        <p className="brief-kicker">How to read this</p>
        <p className="brief-lede">
          Hourly demand on the PJM grid — mid-Atlantic and Midwest, about 65 million people.
        </p>
        <ul className="brief-legend">
          <li>
            <i className="swatch actual" /> Observed
          </li>
          <li>
            <i className="swatch p50" /> This forecast
          </li>
          <li>
            <i className="swatch band" /> P10–P90
          </li>
          {pjm ? (
            <li>
              <i className="swatch benchmark" /> PJM official
            </li>
          ) : null}
        </ul>
      </div>
      <FactStrip facts={facts} />
      <p className={`brief-foot${stale ? " brief-foot-stale" : ""}`}>
        <span>
          Built {published}
          {published ? " · " : ""}
          {formatEasternShort(live.generated_at)}
        </span>
        {stale ? (
          <span className="brief-stale" role="status">
            Older than 12 hours — last successful run, not a live ops feed.
          </span>
        ) : null}
      </p>
    </section>
  );
}

function BacktestBrief({ backtest }: { backtest: BacktestSnapshot | null }) {
  if (!backtest) return null;
  const built = formatEasternParts(backtest.generated_at);
  const hours = backtest.lead_hours;
  const facts: Fact[] = [
    {
      label: "Compared at",
      value: `${hours}h ahead`,
      hint: "Forecast issued this far before the hour being scored",
    },
    {
      label: "Scorecard built",
      value: built.time,
      detail: built.date,
      hint: "When this walk-forward evaluation was written",
    },
  ];

  return (
    <section className="brief" aria-label="How to read this backtest">
      <div className="brief-top">
        <p className="brief-kicker">How to read this</p>
        <p className="brief-lede">
          Same models, scored after the fact. MAE is average error; a calibrated band should
          cover about 80% of actuals.
        </p>
        <ul className="brief-legend">
          <li>
            <i className="swatch actual" /> Actual load
          </li>
          <li>
            <i className="swatch p50" /> Issued forecast
          </li>
          <li>
            <i className="swatch band" /> P10–P90
          </li>
        </ul>
      </div>
      <FactStrip facts={facts} />
    </section>
  );
}

function FactStrip({ facts }: { facts: Fact[] }) {
  return (
    <dl className="brief-facts">
      {facts.map((fact) => (
        <div key={fact.label}>
          <dt>{fact.label}</dt>
          <dd className="brief-value">{fact.value}</dd>
          {fact.detail ? <dd className="brief-date">{fact.detail}</dd> : null}
          <dd className="brief-hint">{fact.hint}</dd>
        </div>
      ))}
    </dl>
  );
}
