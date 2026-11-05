export type Series = { label: string; points: { x: number; y: number }[] };

const W = 560;
const H = 240;
const PAD = { l: 48, r: 12, t: 12, b: 32 };
const COLORS = ["#1d4ed8", "#d97706", "#059669", "#dc2626", "#7c3aed"];

// Small dependency-free SVG line chart, rendered on the server.
export function LineChart({
  title,
  series,
  yLabel,
  yMin,
  yMax,
}: {
  title: string;
  series: Series[];
  yLabel: string;
  yMin?: number;
  yMax?: number;
}) {
  const points = series.flatMap((s) => s.points);
  if (!points.length) return null;
  const xs = points.map((p) => p.x);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const lo = yMin ?? Math.min(0, ...points.map((p) => p.y));
  const hi = yMax ?? (Math.max(...points.map((p) => p.y)) * 1.05 || 1);
  const sx = (x: number) => PAD.l + (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0)) * (W - PAD.l - PAD.r);
  const sy = (y: number) => H - PAD.b - ((y - lo) / (hi - lo || 1)) * (H - PAD.t - PAD.b);
  const ticks = [lo, lo + (hi - lo) / 2, hi];
  const rounds = [...new Set(xs)].sort((a, b) => a - b);

  return (
    <figure className="space-y-1">
      <figcaption className="text-sm font-medium">{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title} className="w-full max-w-xl">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={sy(t)} y2={sy(t)} stroke="currentColor" opacity={0.12} />
            <text x={PAD.l - 6} y={sy(t) + 4} textAnchor="end" fontSize={10} fill="currentColor">
              {Number.isInteger(t) ? t : t.toFixed(2)}
            </text>
          </g>
        ))}
        {rounds.map((r) => (
          <text key={r} x={sx(r)} y={H - 12} textAnchor="middle" fontSize={10} fill="currentColor">
            {r}
          </text>
        ))}
        <text x={PAD.l} y={H - 1} fontSize={10} fill="currentColor" opacity={0.6}>
          round | {yLabel}
        </text>
        {series.map((s, i) => (
          <g key={s.label} stroke={COLORS[i % COLORS.length]} fill={COLORS[i % COLORS.length]}>
            <polyline
              fill="none"
              strokeWidth={2}
              points={s.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")}
            />
            {s.points.map((p) => (
              <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r={3} />
            ))}
          </g>
        ))}
      </svg>
      <div className="flex flex-wrap gap-3 text-xs">
        {series.map((s, i) => (
          <span key={s.label} className="flex items-center gap-1">
            <span className="inline-block h-2 w-3" style={{ background: COLORS[i % COLORS.length] }} />
            {s.label}
          </span>
        ))}
      </div>
    </figure>
  );
}
