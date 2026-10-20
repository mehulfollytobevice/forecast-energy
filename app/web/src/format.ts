export function formatMw(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 10_000) {
    return `${(value / 1000).toFixed(2)} GW`;
  }
  return `${Math.round(value).toLocaleString()} MW`;
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
