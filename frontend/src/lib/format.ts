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
