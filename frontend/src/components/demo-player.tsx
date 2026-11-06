"use client";

import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { buildFrames, type ClientState, type DemoRun } from "@/lib/demo-frames";

const W = 560;
const NODE_COLORS: Record<ClientState, string> = {
  idle: "#9aa3b2",
  training: "#d97706",
  sent: "#059669",
  dropout: "#dc2626",
};
const STATE_LABEL: Record<ClientState, string> = {
  idle: "waiting",
  training: "training",
  sent: "update sent",
  dropout: "no answer",
};
const COORDINATOR_LABEL = {
  waiting: "waiting",
  collecting: "collecting updates",
  down: "stopped",
  aggregating: "aggregating",
  done: "model updated",
} as const;

function Curve({ demo, completed }: { demo: DemoRun; completed: number }) {
  const w = 520;
  const h = 200;
  const pad = { l: 36, r: 10, t: 10, b: 26 };
  const n = demo.rounds.length;
  const sx = (round: number) => pad.l + ((round - 1) / Math.max(1, n - 1)) * (w - pad.l - pad.r);
  const sy = (v: number) => h - pad.b - v * (h - pad.t - pad.b);
  const shown = demo.rounds.slice(0, completed);
  const lines = [
    { label: "F1", color: "#1d4ed8", pick: (r: DemoRun["rounds"][number]) => r.f1 },
    { label: "ROC-AUC", color: "#d97706", pick: (r: DemoRun["rounds"][number]) => r.rocAuc },
  ];
  return (
    <figure className="space-y-2">
      <figcaption className="font-medium">Global model quality after each round</figcaption>
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="F1 and ROC-AUC of the global model by round" className="w-full">
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={w - pad.r} y1={sy(t)} y2={sy(t)} stroke="currentColor" opacity={0.12} />
            <text x={pad.l - 6} y={sy(t) + 4} textAnchor="end" fontSize={10} fill="currentColor">
              {t}
            </text>
          </g>
        ))}
        {demo.rounds.map((r) => (
          <text key={r.round} x={sx(r.round)} y={h - 8} textAnchor="middle" fontSize={10} fill="currentColor">
            {r.round}
          </text>
        ))}
        {lines.map((l) => (
          <g key={l.label} stroke={l.color} fill={l.color}>
            <polyline fill="none" strokeWidth={2} points={shown.map((r) => `${sx(r.round)},${sy(l.pick(r))}`).join(" ")} />
            {shown.map((r) => (
              <circle key={r.round} cx={sx(r.round)} cy={sy(l.pick(r))} r={3} />
            ))}
          </g>
        ))}
      </svg>
      <div className="flex gap-4 text-xs">
        {lines.map((l) => (
          <span key={l.label} className="flex items-center gap-1">
            <span className="inline-block h-2 w-3" style={{ background: l.color }} />
            {l.label}
          </span>
        ))}
      </div>
    </figure>
  );
}

const MOTION_QUERY = "(prefers-reduced-motion: reduce)";

// True when the visitor's system asks for less motion. Assumed on the server, so the replay
// never starts moving before we know.
function useReducedMotion(): boolean {
  return useSyncExternalStore(
    (notify) => {
      const query = window.matchMedia(MOTION_QUERY);
      query.addEventListener("change", notify);
      return () => query.removeEventListener("change", notify);
    },
    () => window.matchMedia(MOTION_QUERY).matches,
    () => true,
  );
}

export function DemoPlayer({ demo }: { demo: DemoRun }) {
  const frames = useMemo(() => buildFrames(demo), [demo]);
  const last = frames.length - 1;
  const [index, setIndex] = useState(0);
  // null: the visitor has not pressed anything yet. Then the replay plays by itself, unless
  // their system asks for less motion.
  const [choice, setChoice] = useState<boolean | null>(null);
  const [speed, setSpeed] = useState(1);
  const reduced = useReducedMotion();
  const playing = (choice ?? !reduced) && index < last;

  useEffect(() => {
    if (!playing) return;
    const timer = setTimeout(() => setIndex((i) => Math.min(last, i + 1)), 1100 / speed);
    return () => clearTimeout(timer);
  }, [playing, index, last, speed]);

  const frame = frames[index];
  const count = demo.clients.length;
  const slot = (i: number) => (W / count) * (i + 0.5);
  const coordinatorDown = frame.coordinator === "down";
  const button = "rounded-lg border border-border px-3 py-1.5 text-sm hover:bg-surface disabled:opacity-40";

  return (
    <div className="space-y-6">
      <div className="space-y-4 rounded-xl border border-border bg-surface p-4">
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          className="rounded-lg bg-accent px-4 py-1.5 text-sm font-medium text-white hover:opacity-90"
          onClick={() => {
            if (index >= last) {
              setIndex(0);
              setChoice(true);
            } else {
              setChoice(!playing);
            }
          }}
        >
          {playing ? "Pause" : index >= last ? "Replay" : "Play"}
        </button>
        <button type="button" className={button} disabled={index >= last} onClick={() => { setChoice(false); setIndex((i) => Math.min(last, i + 1)); }}>
          Next step
        </button>
        <button type="button" className={button} onClick={() => { setChoice(false); setIndex(0); }}>
          Restart
        </button>
        <label className="flex items-center gap-2 text-sm">
          Speed
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="rounded border border-border bg-transparent px-1 py-1">
            <option value={0.5}>0.5x</option>
            <option value={1}>1x</option>
            <option value={2}>2x</option>
          </select>
        </label>
        <span className="ml-auto rounded-full bg-accent-soft px-3 py-1 text-sm font-medium text-accent">
          Round {frame.round} of {demo.rounds.length}
        </span>
      </div>

      <input
        type="range"
        min={0}
        max={last}
        value={index}
        onChange={(e) => { setChoice(false); setIndex(Number(e.target.value)); }}
        aria-label="Position in the replay"
        className="w-full accent-[var(--accent)]"
      />
      </div>

      <p aria-live="polite" className="min-h-16 rounded-xl border-l-4 border-accent bg-accent-soft p-4 text-base">
        {frame.caption}
      </p>

      <div className="grid items-start gap-8 lg:grid-cols-2">
      <svg viewBox={`0 0 ${W} 250`} role="img" aria-label="The coordinator and the clients" className="w-full">
        {demo.clients.map((c, i) => (
          <line
            key={c}
            x1={slot(i)}
            y1={170}
            x2={W / 2}
            y2={70}
            stroke={NODE_COLORS[frame.clients[c]]}
            strokeWidth={frame.clients[c] === "sent" ? 3 : 2}
            strokeDasharray={frame.clients[c] === "sent" ? undefined : "5 5"}
            opacity={coordinatorDown ? 0.3 : 1}
          />
        ))}
        {!reduced && frame.kind === "report" && frame.client && (
          <circle key={index} r={6} fill="#059669">
            <animateMotion
              dur="0.9s"
              fill="freeze"
              path={`M${slot(demo.clients.indexOf(frame.client))},170 L${W / 2},70`}
            />
          </circle>
        )}

        <g>
          <rect x={W / 2 - 80} y={10} width={160} height={60} rx={8} fill="none" stroke={coordinatorDown ? "#dc2626" : "currentColor"} strokeWidth={2} />
          <text x={W / 2} y={34} textAnchor="middle" fontSize={14} fontWeight={600} fill="currentColor">
            Coordinator
          </text>
          <text x={W / 2} y={55} textAnchor="middle" fontSize={11} fill={coordinatorDown ? "#dc2626" : "currentColor"}>
            {COORDINATOR_LABEL[frame.coordinator]}
          </text>
        </g>
        <g opacity={0.85}>
          <rect x={W / 2 + 110} y={20} width={110} height={40} rx={6} fill="none" stroke="currentColor" strokeDasharray="3 3" />
          <text x={W / 2 + 165} y={44} textAnchor="middle" fontSize={11} fill="currentColor">
            Database
          </text>
          <line x1={W / 2 + 80} y1={40} x2={W / 2 + 110} y2={40} stroke="currentColor" strokeDasharray="3 3" />
        </g>

        {demo.clients.map((c, i) => (
          <g key={c}>
            <rect x={slot(i) - 45} y={170} width={90} height={56} rx={8} fill="none" stroke={NODE_COLORS[frame.clients[c]]} strokeWidth={2.5} />
            <text x={slot(i)} y={192} textAnchor="middle" fontSize={12} fontWeight={600} fill="currentColor">
              {c}
            </text>
            <text x={slot(i)} y={212} textAnchor="middle" fontSize={10} fill={NODE_COLORS[frame.clients[c]]}>
              {STATE_LABEL[frame.clients[c]]}
            </text>
          </g>
        ))}
      </svg>

      <Curve demo={demo} completed={frame.completed} />
      </div>

      <details className="rounded-xl border border-border text-sm">
        <summary className="cursor-pointer px-4 py-3 font-medium">Results of every round as a table</summary>
        <table className="w-full border-t border-border text-left">
          <thead>
            <tr className="border-b border-border text-muted">
              <th className="px-2 py-1">Round</th>
              <th className="px-2 py-1">Clients that reported</th>
              <th className="px-2 py-1">F1</th>
              <th className="px-2 py-1">ROC-AUC</th>
              <th className="px-2 py-1">PR-AUC</th>
            </tr>
          </thead>
          <tbody>
            {demo.rounds.map((r) => (
              <tr key={r.round} className="border-b border-border">
                <td className="px-2 py-1">{r.round}</td>
                <td className="px-2 py-1">
                  {r.participants.filter((p) => p.status === "UPDATE_ACCEPTED").length} of {r.participants.length}
                </td>
                <td className="px-2 py-1">{r.f1.toFixed(3)}</td>
                <td className="px-2 py-1">{r.rocAuc.toFixed(3)}</td>
                <td className="px-2 py-1">{r.prAuc.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
