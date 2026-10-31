const EASTERN = "America/New_York";

export function formatMw(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 10_000) {
    return `${(value / 1000).toFixed(2)} GW`;
  }
  return `${Math.round(value).toLocaleString()} MW`;
}

/** Full Eastern timestamp for chart hover, e.g. "Thu, Oct 30, 2026 · 10:00 PM ET". */
export function formatHoverTime(iso: string): string {
  const d = new Date(iso);
  const date = d.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: EASTERN,
  });
  const time = d.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
    timeZone: EASTERN,
  });
  return `${date} · ${time} ET`;
}

export function formatEasternParts(iso: string): { date: string; time: string } {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return { date: iso, time: "" };
  const date = d.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    timeZone: EASTERN,
  });
  const time = d.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
    timeZone: EASTERN,
  });
  return { date, time: `${time} ET` };
}

/** Compact Eastern clock, e.g. "Fri, Oct 30 · 6:00 PM ET". */
export function formatEasternShort(iso: string): string {
  const { date, time } = formatEasternParts(iso);
  return time ? `${date} · ${time}` : date;
}

export function formatRelativeTime(iso: string, nowMs: number = Date.now()): string {
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return "";
  const minutes = Math.round((nowMs - then) / 60_000);
  const abs = Math.abs(minutes);
  const suffix = minutes < 0 ? "from now" : "ago";
  if (abs < 1) return "just now";
  if (abs < 60) return `${abs} min ${suffix}`;
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 36) return `${Math.abs(hours)}h ${hours < 0 ? "from now" : "ago"}`;
  const days = Math.round(hours / 24);
  return `${Math.abs(days)}d ${days < 0 ? "from now" : "ago"}`;
}

/** Decode `202610301200Z` / `20260601T0600Z` into a readable Eastern time. */
export function formatWeatherRun(run: string): { date: string; time: string; code: string } {
  const match = run.trim().match(/^(\d{4})(\d{2})(\d{2})T?(\d{2})(\d{2})Z$/i);
  if (!match) return { date: run, time: "", code: run };
  const iso = `${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:00Z`;
  return { ...formatEasternParts(iso), code: run };
}

export function nodeKey(id: string, level: string): string {
  return `${level}:${id}`;
}

/** Interpolate between two hex colors (0–1). */
export function lerpColor(t: number, low: string, high: string): string {
  const parse = (hex: string) => {
    const h = hex.replace("#", "");
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  };
  const a = parse(low);
  const b = parse(high);
  const mix = (i: number) => Math.round(a[i] + t * (b[i] - a[i]));
  return `rgb(${mix(0)}, ${mix(1)}, ${mix(2)})`;
}
