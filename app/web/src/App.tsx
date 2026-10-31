import type { FeatureCollection } from "geojson";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { BacktestChart } from "./BacktestChart";
import { BacktestDateFilter } from "./BacktestDateFilter";
import {
  clampDateWindow,
  nodeDateRange,
  sliceBacktestNodes,
  unionDateRange,
  type DateWindow,
} from "./backtestWindow";
import { FanChart } from "./FanChart";
import { ForecastMap } from "./ForecastMap";
import { SnapshotBrief } from "./SnapshotBrief";
import { nodeKey } from "./format";
import {
  buildNodeIndex,
  cardsForFocus,
  defaultPath,
  drillTarget,
  emptyCollection,
  pathToFocus,
  truncatePath,
} from "./hierarchy";
import type {
  BacktestNode,
  BacktestSnapshot,
  DashboardTab,
  Focus,
  ForecastNode,
  ForecastSnapshot,
  GeoProperties,
} from "./types";

const STALE_AFTER_MS = 12 * 60 * 60 * 1000;

function isStaleForecast(generatedAt: string, nowMs: number = Date.now()): boolean {
  const generated = Date.parse(generatedAt);
  if (Number.isNaN(generated)) return false;
  return nowMs - generated > STALE_AFTER_MS;
}

export function App() {
  const [tab, setTab] = useState<DashboardTab>("live");
  const [live, setLive] = useState<ForecastSnapshot | null>(null);
  const [backtest, setBacktest] = useState<BacktestSnapshot | null>(null);
  const [geo, setGeo] = useState<FeatureCollection>(emptyCollection());
  const [path, setPath] = useState<Focus[]>(defaultPath);
  const [error, setError] = useState<string | null>(null);
  const [backtestError, setBacktestError] = useState<string | null>(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [dateWindow, setDateWindow] = useState<DateWindow | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [forecastRes, geoRes] = await Promise.all([
          fetch("/api/forecast"),
          fetch("/geo/pjm_regions.geojson"),
        ]);
        if (!forecastRes.ok) throw new Error(`forecast ${forecastRes.status}`);
        if (!geoRes.ok) throw new Error(`geo ${geoRes.status}`);
        const forecast = (await forecastRes.json()) as ForecastSnapshot;
        const collection = (await geoRes.json()) as FeatureCollection;
        if (!cancelled) {
          setLive(forecast);
          setGeo(collection);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "load failed");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (tab !== "backtest" || backtest) return;
    let cancelled = false;
    setBacktestLoading(true);
    setBacktestError(null);
    (async () => {
      try {
        const response = await fetch("/api/backtest");
        if (!response.ok) throw new Error(`backtest ${response.status}`);
        const payload = (await response.json()) as BacktestSnapshot;
        if (!cancelled) setBacktest(payload);
      } catch (err) {
        if (!cancelled) {
          setBacktestError(err instanceof Error ? err.message : "backtest load failed");
        }
      } finally {
        if (!cancelled) setBacktestLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [tab, backtest]);

  const selectTab = useCallback((next: DashboardTab) => {
    setTab(next);
    setPath(defaultPath());
  }, []);

  const focus = pathToFocus(path);
  const liveIndex = useMemo(() => (live ? buildNodeIndex(live.nodes) : null), [live]);
  const backtestIndex = useMemo(
    () => (backtest ? buildNodeIndex(backtest.nodes) : null),
    [backtest],
  );
  const liveCards = useMemo(
    () => (liveIndex ? cardsForFocus(geo, focus, liveIndex) : []),
    [liveIndex, geo, focus],
  );
  const backtestCardsRaw = useMemo(
    () => (backtestIndex ? cardsForFocus(geo, focus, backtestIndex) : []),
    [backtestIndex, geo, focus],
  );
  const availableRange = useMemo(
    () => unionDateRange(backtestCardsRaw),
    [backtestCardsRaw],
  );
  const availableKeyRef = useRef<string | null>(null);

  useEffect(() => {
    if (!availableRange) {
      availableKeyRef.current = null;
      setDateWindow(null);
      return;
    }
    const key = `${availableRange.start}|${availableRange.end}`;
    const boundsChanged = availableKeyRef.current !== key;
    availableKeyRef.current = key;
    setDateWindow((prev) => {
      // New hierarchy view → show that view's full available span.
      if (!prev || boundsChanged) return { ...availableRange };
      const next = clampDateWindow(prev, availableRange);
      if (prev.start === next.start && prev.end === next.end) return prev;
      return next;
    });
  }, [availableRange]);

  const windowedNodes = useMemo(() => {
    if (!backtest || !dateWindow) return [];
    return sliceBacktestNodes(backtest.nodes, dateWindow);
  }, [backtest, dateWindow]);

  const windowedIndex = useMemo(
    () => (windowedNodes.length ? buildNodeIndex(windowedNodes) : null),
    [windowedNodes],
  );
  const backtestCards = useMemo(
    () => (windowedIndex ? cardsForFocus(geo, focus, windowedIndex) : []),
    [windowedIndex, geo, focus],
  );

  const onDrill = useCallback(
    (props: GeoProperties) => {
      const next = drillTarget(focus, props);
      if (!next) return;
      setPath((current) => [...current, next]);
    },
    [focus],
  );

  const onSelectLiveCard = useCallback(
    (node: ForecastNode) => {
      if (node.level === "load_area") return;
      if (focus.level === "zone" && node.id === focus.id && node.level === "zone") {
        setPath((current) => [
          ...current,
          { id: node.id, level: "load_area", name: node.name },
        ]);
        return;
      }
      if (node.id === focus.id && node.level === focus.level) return;
      setPath((current) => [
        ...current,
        { id: node.id, level: node.level, name: node.name },
      ]);
    },
    [focus],
  );

  const onSelectBacktestCard = useCallback(
    (node: BacktestNode) => {
      if (node.level === "load_area") return;
      if (focus.level === "zone" && node.id === focus.id && node.level === "zone") {
        setPath((current) => [
          ...current,
          { id: node.id, level: "load_area", name: node.name },
        ]);
        return;
      }
      if (node.id === focus.id && node.level === focus.level) return;
      setPath((current) => [
        ...current,
        { id: node.id, level: node.level, name: node.name },
      ]);
    },
    [focus],
  );

  if (error) {
    return (
      <main className="page">
        <p className="error">Could not load dashboard: {error}</p>
      </main>
    );
  }
  if (!live) {
    return (
      <main className="page">
        <p className="loading">Loading forecast…</p>
      </main>
    );
  }

  const tagline =
    tab === "live"
      ? "Next 48 hours of demand, with uncertainty bands and PJM’s official forecast as a baseline. Click the map or a chart title to drill from the full grid to a utility."
      : "Issued forecasts versus what load actually was. Pick a date range; totals roll up from load areas to zones, regions, and PJM.";
  const stale = isStaleForecast(live.generated_at);

  return (
    <main className="page">
      <header className="site-header">
        <div className="brand-block">
          <p className="eyebrow">PJM Interconnection · short-term electricity demand</p>
          <h1 className="brand">
            <span className="brand-mark">PJM</span>
            <span className="brand-rest">Load forecast</span>
          </h1>
          <p className="tagline">{tagline}</p>
        </div>

        <div className="tab-bar" role="tablist" aria-label="Dashboard mode">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "live"}
            className={tab === "live" ? "tab active" : "tab"}
            onClick={() => selectTab("live")}
          >
            Live forecast
            <span className="tab-hint">Next 48 hours</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "backtest"}
            className={tab === "backtest" ? "tab active" : "tab"}
            onClick={() => selectTab("backtest")}
          >
            Backtest
            <span className="tab-hint">Forecast vs actual</span>
          </button>
        </div>

        <SnapshotBrief tab={tab} live={live} backtest={backtest} stale={stale} />

        <nav className="breadcrumb" aria-label="Hierarchy">
          {path.map((crumb, index) => (
            <span key={`${crumb.level}:${crumb.id}`}>
              {index > 0 && <span className="breadcrumb-sep" aria-hidden="true">/</span>}
              <button type="button" onClick={() => setPath(truncatePath(path, index))}>
                {crumb.name}
              </button>
            </span>
          ))}
        </nav>
      </header>

      {tab === "live" ? (
        <>
          <ForecastMap
            geo={geo}
            focus={focus}
            nodes={live.nodes}
            onDrill={onDrill}
            colorBy="p50"
          />
          <p className="section-label">Live region forecasts · next 48 hours</p>
          <section className="cards" aria-label="Live region forecasts">
            {liveCards.map((node) => (
              <FanChart
                key={`${node.level}:${node.id}`}
                node={node}
                onSelect={node.level === "load_area" ? undefined : onSelectLiveCard}
              />
            ))}
          </section>
        </>
      ) : backtestLoading && !backtest ? (
        <p className="loading">Loading backtest…</p>
      ) : backtestError ? (
        <p className="error">Could not load backtest: {backtestError}</p>
      ) : backtest && availableRange && dateWindow ? (
        <>
          <BacktestDateFilter
            available={availableRange}
            value={dateWindow}
            onChange={setDateWindow}
          />
          <ForecastMap
            geo={geo}
            focus={focus}
            nodes={windowedNodes}
            onDrill={onDrill}
            colorBy="mae"
          />
          <p className="section-label">Backtest · actual vs forecast</p>
          <section className="cards" aria-label="Backtest region series">
            {backtestCards.map((node) => {
              const raw = backtestIndex?.get(nodeKey(node.id, node.level));
              const available = raw ? nodeDateRange(raw) : null;
              return (
                <BacktestChart
                  key={nodeKey(node.id, node.level)}
                  node={node}
                  windowStart={dateWindow.start}
                  windowEnd={dateWindow.end}
                  availableStart={available?.start}
                  availableEnd={available?.end}
                  leadHours={backtest.lead_hours}
                  onSelect={node.level === "load_area" ? undefined : onSelectBacktestCard}
                />
              );
            })}
          </section>
        </>
      ) : null}
    </main>
  );
}
