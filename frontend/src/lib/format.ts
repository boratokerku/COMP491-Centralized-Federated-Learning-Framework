export function fmtTime(value: Date | string | null | undefined): string {
  if (!value) return "-";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? "-" : d.toISOString().replace("T", " ").slice(0, 19) + " UTC";
}

export function fmtNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined) return "-";
  return Number.isInteger(value) ? value.toLocaleString("en-US") : value.toFixed(digits);
}

export function fmtBytes(value: number | null | undefined): string {
  if (value === null || value === undefined) return "-";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(2)} MB`;
}

export const shortId = (id: string) => id.slice(0, 8);

/** "2 d 3 h", "4 h 05 min", "3:07" for a time left until a deadline. */
export function fmtRemaining(ms: number): string {
  if (ms <= 0) return "deadline passed";
  const total = Math.round(ms / 1000);
  const d = Math.floor(total / 86400);
  const h = Math.floor((total % 86400) / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (d > 0) return `${d} d ${h} h`;
  if (h > 0) return `${h} h ${String(m).padStart(2, "0")} min`;
  return `${m}:${String(s).padStart(2, "0")}`;
}
