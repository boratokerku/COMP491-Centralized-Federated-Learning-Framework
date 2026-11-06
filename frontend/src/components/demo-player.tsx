"use client";

import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type CSSProperties } from "react";
import {
  buildFrames,
  buildScenarios,
  COORDINATOR,
  DATABASE,
  GATEWAY,
  type ClientState,
  type CoordinatorState,
  type DemoRun,
  type Frame,
  type Packet,
  type Scenario,
} from "@/lib/demo-frames";

// Text keeps its size in the viewBox, so a wider stage means more air, not bigger boxes.
const WIDE_STAGE = {
  W: 760,
  H: 444,
  COORD: { x: 250, y: 18, w: 200, h: 88 },
  DB: { x: 556, y: 28, w: 190, h: 68 },
  GATE: { x: 14, y: 186, w: 732, h: 44 },
  CLIENT_TOP: 322,
  boxW: 140,
  boxH: 64,
};
const NARROW_STAGE = {
  W: 420,
  H: 360,
  COORD: { x: 110, y: 14, w: 140, h: 76 },
  DB: { x: 300, y: 26, w: 112, h: 50 },
  GATE: { x: 6, y: 138, w: 408, h: 36 },
  CLIENT_TOP: 252,
  boxW: 92,
  boxH: 54,
};

const GREEN = "#059669";
const BLUE = "var(--accent)";
const RED = "var(--danger)";
const AMBER = "var(--warning)";
const NODE_COLORS: Record<ClientState, string> = {
  idle: "var(--muted)",
  checking: BLUE,
  blocked: RED,
  training: AMBER,
  sending: BLUE,
  sent: GREEN,
  dropout: RED,
  late: RED,
  rejected: RED,
};
const STATE_LABEL: Record<ClientState, string> = {
  idle: "waiting",
  checking: "being checked",
  blocked: "not eligible",
  training: "training",
  sending: "sending",
  sent: "update sent",
  dropout: "no answer",
  late: "too late",
  rejected: "excluded",
};
const COORDINATOR_LABEL: Record<CoordinatorState, string> = {
  waiting: "waiting",
  checking: "checking",
  opening: "opening the round",
  collecting: "collecting updates",
  down: "stopped",
  restarting: "restarting",
  aggregating: "aggregating",
  done: "model updated",
  failed: "round failed",
};

const vars = (values: Record<string, string>) => values as CSSProperties;

function Curve({ demo, completed }: { demo: DemoRun; completed: number }) {
  const w = 360;
  const h = 220;
  const pad = { l: 28, r: 10, t: 10, b: 26 };
  const n = demo.rounds.length;
  const sx = (round: number) => pad.l + ((round - 1) / Math.max(1, n - 1)) * (w - pad.l - pad.r);
  const sy = (v: number) => h - pad.b - v * (h - pad.t - pad.b);
  const shown = demo.rounds.slice(0, completed);
  const latest = shown.at(-1);
  const lines = [
    { label: "F1", color: BLUE, pick: (r: DemoRun["rounds"][number]) => r.f1 },
    { label: "PR-AUC", color: AMBER, pick: (r: DemoRun["rounds"][number]) => r.prAuc },
  ];
  return (
    <figure className="space-y-2">
      <figcaption className="font-medium">Global model quality after each round</figcaption>
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="F1 and PR-AUC of the global model by round" className="w-full">
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={w - pad.r} y1={sy(t)} y2={sy(t)} stroke="currentColor" opacity={0.12} />
            <text x={pad.l - 6} y={sy(t) + 4} textAnchor="end" fontSize={11} fill="currentColor">
              {t}
            </text>
          </g>
        ))}
        {demo.rounds.map((r) => (
          <text key={r.round} x={sx(r.round)} y={h - 8} textAnchor="middle" fontSize={11} fill="currentColor">
            {r.round}
          </text>
        ))}
        {lines.map((l) => (
          <g key={l.label}>
            {shown.slice(1).map((r, i) => (
              <line
                key={r.round}
                x1={sx(shown[i].round)}
                y1={sy(l.pick(shown[i]))}
                x2={sx(r.round)}
                y2={sy(l.pick(r))}
                stroke={l.color}
                strokeWidth={2.5}
                pathLength={1}
                className={r.round === completed ? "demo-draw" : undefined}
              />
            ))}
            {shown.map((r) => (
              <circle
                key={r.round}
                cx={sx(r.round)}
                cy={sy(l.pick(r))}
                r={3.5}
                fill={l.color}
                className={r.round === completed ? "demo-pop" : undefined}
              />
            ))}
          </g>
        ))}
      </svg>
      <div className="flex gap-4 text-sm">
        {lines.map((l) => (
          <span key={l.label} className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-3 rounded-sm" style={{ background: l.color }} />
            {l.label}
            {latest && <span className="font-medium tabular-nums">{l.pick(latest).toFixed(2)}</span>}
          </span>
        ))}
      </div>
    </figure>
  );
}

function ScenarioPanel({ scenario, index }: { scenario: Scenario; index: number }) {
  const events = scenario.frames.slice(0, index + 1).flatMap((f, i) => (f.event ? [{ ...f.event, i }] : []));
  const finished = index === scenario.frames.length - 1;
  return (
    <div className="space-y-4">
      <div>
        <h2 className="font-semibold">{scenario.title}</h2>
        <p className="mt-1 text-sm text-muted">{scenario.problem}</p>
      </div>
      <div className="rounded-lg border border-border bg-surface p-3">
        <h3 className="text-xs font-medium text-muted">What the system logs</h3>
        <ol className="mt-2 min-h-32 space-y-1 font-mono text-xs">
          {events.map((e) => (
            <li key={e.i} className={e.i === index ? "demo-fade break-words font-semibold" : "break-words text-muted"}>
              {e.name}
              {e.detail && <span className="font-normal"> · {e.detail}</span>}
            </li>
          ))}
        </ol>
      </div>
      {finished && (
        <p className="demo-fade rounded-lg border-l-4 p-3 text-sm" style={{ borderColor: GREEN }}>
          <strong>Result:</strong> {scenario.outcome}
        </p>
      )}
    </div>
  );
}

function PacketShape({ packet }: { packet: Packet }) {
  const text = (value: string, width: number) => (
    <>
      <rect x={-width / 2} y={-8} width={width} height={16} rx={4} fill="var(--accent)" />
      <text textAnchor="middle" y={4} fontSize={10} fontWeight={600} fill="var(--background)">
        {value}
      </text>
    </>
  );
  switch (packet.kind) {
    case "model":
      return text(packet.label ?? "model", 30);
    case "state":
      return text("state", 36);
    case "updates":
      return text("updates", 50);
    case "report":
      return text("report", 40);
    case "ack":
      return (
        <>
          <circle r={8} fill={GREEN} />
          <path d="M-3.5 0 L-1 2.5 L3.5 -2.5" fill="none" stroke="white" strokeWidth={1.8} />
        </>
      );
    case "broken":
      return (
        <>
          <circle r={8} fill={RED} />
          <text textAnchor="middle" y={4} fontSize={12} fontWeight={700} fill="var(--background)">
            !
          </text>
        </>
      );
    default:
      return <circle r={7} fill={packet.fate ? RED : GREEN} />;
  }
}

// One moving packet along a route of straight pieces. A refused or lost one stops short.
function MovingPacket({ packet, route, ms }: { packet: Packet; route: Point[]; ms: number }) {
  const reach = packet.fate === "refused" ? 0.8 : packet.fate === "lost" ? 0.55 : 1;
  const stops = route.slice(0, -1);
  const [a, b] = route.slice(-2);
  stops.push({ x: a.x + (b.x - a.x) * reach, y: a.y + (b.y - a.y) * reach });
  const path = stops.map((p, i) => `${i === 0 ? "M" : "L"}${p.x} ${p.y}`).join(" ");
  return (
    <g className="demo-packet" style={{ offsetPath: `path("${path}")`, ...vars({ "--dur": `${ms}ms` }) }}>
      <PacketShape packet={packet} />
    </g>
  );
}

type Point = { x: number; y: number };

function Stage({ frame, clients, animate, ms, id }: { frame: Frame; clients: string[]; animate: boolean; ms: number; id: string }) {
  const { W, H, COORD, DB, GATE, CLIENT_TOP, boxW, boxH } = useMatchMedia(WIDE_QUERY, true) ? WIDE_STAGE : NARROW_STAGE;
  const count = clients.length;
  const slot = (i: number) => (W / count) * (i + 0.5);
  const coordX = COORD.x + COORD.w / 2;
  const down = frame.coordinator === "down";
  const failed = frame.coordinator === "failed";
  const coordinatorColor = down || failed ? RED : frame.coordinator === "done" ? GREEN : "currentColor";
  const linkY = DB.y + DB.h / 2;
  const dbX = DB.x + DB.w / 2;
  const gateY = GATE.y + GATE.h / 2;

  // Clients reach the database only through the gateway.
  const route = ({ from, to }: Packet): Point[] => {
    if (from === COORDINATOR) return [{ x: COORD.x + COORD.w, y: linkY }, { x: DB.x, y: linkY }];
    if (to === COORDINATOR) return [{ x: DB.x, y: linkY }, { x: COORD.x + COORD.w, y: linkY }];
    const client = slot(clients.indexOf(from === DATABASE || from === GATEWAY ? to : from));
    const door = [{ x: client, y: CLIENT_TOP }, { x: client, y: gateY }];
    const full = to === GATEWAY || from === GATEWAY ? door : [...door, { x: dbX, y: gateY }, { x: dbX, y: DB.y + DB.h }];
    return from === DATABASE || from === GATEWAY ? full.toReversed() : full;
  };
  const touches = (node: string) => frame.packets.some((p) => p.from === node || p.to === node);
  const linked = frame.packets.some((p) => p.from === COORDINATOR || p.to === COORDINATOR);
  const database = frame.database || touches(DATABASE);
  const gateway = frame.packets.some((p) => p.from !== COORDINATOR && p.to !== COORDINATOR);
  const refused = frame.packets.some((p) => p.fate === "refused");
  const gatewayColor = refused ? RED : gateway ? "var(--accent)" : "currentColor";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="The clients reach the database through the gateway; the coordinator works from the database" className="w-full">
      {clients.map((c, i) => {
        const state = frame.clients[c];
        const quiet = state === "blocked" || state === "dropout" || state === "idle";
        return (
          <line
            key={c}
            x1={slot(i)}
            y1={CLIENT_TOP}
            x2={slot(i)}
            y2={GATE.y + GATE.h}
            stroke={NODE_COLORS[state]}
            strokeWidth={state === "sent" ? 2.5 : 1.5}
            strokeDasharray={state === "sent" ? undefined : "4 5"}
            opacity={quiet ? 0.45 : 1}
          />
        );
      })}
      <line
        x1={dbX}
        y1={DB.y + DB.h}
        x2={dbX}
        y2={GATE.y}
        stroke={database && gateway ? "var(--accent)" : "currentColor"}
        strokeDasharray="3 3"
        opacity={database && gateway ? 1 : 0.5}
      />
      <line
        x1={COORD.x + COORD.w}
        y1={linkY}
        x2={DB.x}
        y2={linkY}
        stroke={linked || frame.database ? "var(--accent)" : "currentColor"}
        strokeDasharray="3 3"
        opacity={linked || frame.database ? 1 : 0.5}
      />

      <rect x={GATE.x} y={GATE.y} width={GATE.w} height={GATE.h} rx={8} fill="var(--background)" stroke={gatewayColor} strokeWidth={gateway ? 2 : 1.5} />
      <text x={GATE.x + 14} y={gateY + 4.5} fontSize={13} fontWeight={650} fill="currentColor">
        Gateway
      </text>
      <text x={GATE.x + 92} y={gateY + 4} fontSize={11} fill="currentColor" opacity={0.7}>
        the clients&apos; only way in
      </text>

      <g fill="currentColor">
        <text x={GATE.x} y={34} fontSize={20} fontWeight={650} className="tabular-nums">
          {frame.received}/{frame.expected}
        </text>
        <text x={GATE.x} y={50} fontSize={11}>
          updates
        </text>
        {frame.quorum !== null && (
          <text x={GATE.x} y={65} fontSize={11} opacity={0.7}>
            minimum {frame.quorum}
          </text>
        )}
      </g>

      <rect
        x={COORD.x}
        y={COORD.y}
        width={COORD.w}
        height={COORD.h}
        rx={8}
        fill="var(--background)"
        stroke={coordinatorColor}
        strokeWidth={2}
        className={frame.coordinator === "aggregating" ? "demo-pulse" : undefined}
        opacity={down ? 0.7 : 1}
      />
      <g opacity={down ? 0.6 : 1}>
        <text x={coordX} y={COORD.y + COORD.h * 0.3} textAnchor="middle" fontSize={15} fontWeight={650} fill="currentColor">
          Coordinator
        </text>
        <text key={frame.model} x={coordX} y={COORD.y + COORD.h * 0.56} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--accent)" className="demo-fade">
          global model v{frame.model}
        </text>
        <text x={coordX} y={COORD.y + COORD.h * 0.82} textAnchor="middle" fontSize={11} fill={down || failed ? RED : "currentColor"}>
          {COORDINATOR_LABEL[frame.coordinator]}
        </text>
      </g>

      {frame.deadline !== null && (
        <g>
          <rect x={COORD.x} y={COORD.y + COORD.h + 10} width={COORD.w} height={4} rx={2} fill="currentColor" opacity={0.15} />
          <rect
            x={COORD.x}
            y={COORD.y + COORD.h + 10}
            width={COORD.w}
            height={4}
            rx={2}
            fill={frame.deadline < 0.35 ? RED : AMBER}
            className="demo-bar"
            style={{
              transform: `scaleX(${frame.deadline})`,
              transition: frame.kind === "open" ? "none" : `transform ${ms}ms linear`,
            }}
          />
          <text x={COORD.x - 6} y={COORD.y + COORD.h + 15} textAnchor="end" fontSize={11} fill="currentColor">
            deadline
          </text>
        </g>
      )}

      <rect
        x={DB.x}
        y={DB.y}
        width={DB.w}
        height={DB.h}
        rx={6}
        fill="var(--background)"
        stroke={database ? "var(--accent)" : "currentColor"}
        strokeWidth={database ? 2 : 1}
        strokeDasharray={database ? undefined : "3 3"}
        className={frame.database ? "demo-pulse" : undefined}
      />
      <text x={dbX} y={DB.y + DB.h / 2 - 3} textAnchor="middle" fontSize={13} fontWeight={650} fill="currentColor">
        Database
      </text>
      <text x={dbX} y={DB.y + DB.h / 2 + 13} textAnchor="middle" fontSize={11} fill="currentColor" opacity={0.7}>
        and model files
      </text>

      {clients.map((c, i) => {
        const state = frame.clients[c];
        const color = NODE_COLORS[state];
        const x = slot(i);
        const out = state === "dropout" || state === "late" || state === "rejected" || state === "blocked";
        return (
          <g key={c}>
            <rect x={x - W / count / 2 + 3} y={CLIENT_TOP - 8} width={W / count - 6} height={H - CLIENT_TOP + 6} rx={10} fill="none" stroke="currentColor" strokeDasharray="2 4" opacity={0.3} />
            <rect x={x - boxW / 2} y={CLIENT_TOP} width={boxW} height={boxH} rx={8} fill="var(--background)" stroke={color} strokeWidth={2.5} />
            <text x={x} y={CLIENT_TOP + boxH * 0.36} textAnchor="middle" fontSize={13} fontWeight={650} fill="currentColor">
              {c}
            </text>
            <text x={x} y={CLIENT_TOP + boxH * 0.65} textAnchor="middle" fontSize={11} fill={color}>
              {STATE_LABEL[state]}
            </text>
            {state === "training" && (
              <rect
                key={`${id}-${frame.round}`}
                x={x - boxW / 2 + 8}
                y={CLIENT_TOP + boxH - 12}
                width={boxW - 16}
                height={4}
                rx={2}
                fill={AMBER}
                className={animate ? "demo-fill" : undefined}
                style={vars({ "--dur": `${ms * [1, 0.72, 0.9, 0.82][i % 4]}ms` })}
              />
            )}
            {out && (
              <g transform={`translate(${x + boxW / 2 - 4} ${CLIENT_TOP + 4})`} className="demo-pop">
                <circle r={9} fill={RED} />
                <path d="M-3.5 -3.5 L3.5 3.5 M3.5 -3.5 L-3.5 3.5" stroke="var(--background)" strokeWidth={2} />
              </g>
            )}
            <g transform={`translate(${x - 31} ${CLIENT_TOP + boxH + 14})`} fill="currentColor">
              <path d="M0 3 a6 3 0 0 1 12 0 v8 a6 3 0 0 1 -12 0 Z" fill="none" stroke="currentColor" strokeWidth={1.3} />
              <path d="M0 3 a6 3 0 0 0 12 0" fill="none" stroke="currentColor" strokeWidth={1.3} />
              <text x={18} y={11} fontSize={11}>
                its data
              </text>
            </g>
          </g>
        );
      })}

      {animate &&
        frame.packets.map((p, i) => (
          <MovingPacket key={`${id}-${i}`} packet={p} route={route(p)} ms={ms * 0.9} />
        ))}
    </svg>
  );
}

const MOTION_QUERY = "(prefers-reduced-motion: reduce)";
const WIDE_QUERY = "(min-width: 768px)";

function useMatchMedia(media: string, onServer: boolean): boolean {
  return useSyncExternalStore(
    (notify) => {
      const query = window.matchMedia(media);
      query.addEventListener("change", notify);
      return () => query.removeEventListener("change", notify);
    },
    () => window.matchMedia(media).matches,
    () => onServer,
  );
}

export function DemoPlayer({ demo }: { demo: DemoRun }) {
  const runFrames = useMemo(() => buildFrames(demo), [demo]);
  const scenarios = useMemo(() => buildScenarios(), []);
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const scenario = scenarios.find((s) => s.id === scenarioId) ?? null;
  const frames = scenario?.frames ?? runFrames;
  const last = frames.length - 1;
  const [index, setIndex] = useState(0);
  // null: the visitor has not pressed anything yet. Then the replay plays by itself, unless
  // their system asks for less motion.
  const [choice, setChoice] = useState<boolean | null>(null);
  const [speed, setSpeed] = useState(1);
  // Assumed on the server, so the replay never starts moving before we know.
  const reduced = useMatchMedia(MOTION_QUERY, true);
  const stage = useRef<HTMLDivElement>(null);
  const playing = (choice ?? !reduced) && index < last;

  const frame = frames[index];
  const ms = frame.duration / speed;

  useEffect(() => {
    if (!playing) return;
    const timer = setTimeout(() => setIndex((i) => Math.min(last, i + 1)), ms);
    return () => clearTimeout(timer);
  }, [playing, index, last, ms]);

  const watch = (id: string | null) => {
    setScenarioId(id);
    setIndex(0);
    setChoice(null);
    stage.current?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
  };
  const jump = (i: number) => {
    setChoice(false);
    setIndex(i);
  };
  // While playing, the slider glides to the next step for as long as this step lasts.
  const position = (Math.min(last, index + (playing ? 1 : 0)) / last) * 100;
  const glide = (property: string) =>
    reduced ? "none" : `${property} ${playing ? ms : 150}ms ${playing ? "linear" : "ease-out"}`;
  const chapters = frames.flatMap((f, i) => (f.chapter ? [{ label: f.chapter, i }] : []));
  const button = "rounded-lg border border-border px-3 py-1.5 text-sm hover:bg-surface disabled:opacity-40";
  const picks: { id: string | null; title: string; problem: string }[] = [
    { id: null, title: "The full run", problem: `${demo.rounds.length} rounds, from the first model to the last` },
    ...scenarios,
  ];

  return (
    <div className="space-y-10">
      <div className="space-y-2">
        <h2 className="text-sm font-medium text-muted">Watch the full run, or one failure on its own</h2>
        <div role="group" aria-label="What to watch" className="grid grid-cols-2 gap-2 lg:grid-cols-4">
          {picks.map((p) => {
            const active = p.id === scenarioId;
            return (
              <button
                key={p.id ?? "run"}
                type="button"
                aria-pressed={active}
                onClick={() => watch(p.id)}
                className={`rounded-lg border p-3 text-left text-sm ${active ? "border-accent bg-accent-soft" : "border-border hover:bg-surface"}`}
              >
                <span className={`block font-semibold ${active ? "text-accent" : ""}`}>{p.title}</span>
                <span className="mt-1 hidden text-xs text-muted sm:block">{p.problem}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div ref={stage} className="scroll-mt-24">
        <Stage key={scenarioId ?? "run"} frame={frame} clients={demo.clients} animate={!reduced} ms={ms} id={`${index}`} />
      </div>

      <div className="grid items-start gap-10 lg:grid-cols-[3fr_2fr]">
        <div className="space-y-5">
          <div aria-live="polite" className="min-h-24 rounded-xl border-l-4 border-accent bg-accent-soft p-4">
            <p key={frame.caption} className="demo-fade text-base">
              {frame.caption}
            </p>
            {frame.event && <p className="mt-1 break-words font-mono text-xs text-muted">{frame.event.name}</p>}
          </div>

          <div className="space-y-4 rounded-xl border border-border bg-surface p-4">
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                className="rounded-lg bg-accent px-4 py-1.5 text-sm font-medium text-background hover:opacity-90"
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
              <button type="button" className={button} disabled={index <= 0} onClick={() => jump(index - 1)}>
                Back
              </button>
              <button type="button" className={button} disabled={index >= last} onClick={() => jump(index + 1)}>
                Next step
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
                {scenario ? `Step ${index + 1} of ${frames.length}` : `Round ${frame.round} of ${demo.rounds.length}`}
              </span>
            </div>

            {/* The input stays on top, invisible, for dragging and the keyboard. */}
            <div className="relative h-6">
              <div className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 overflow-hidden rounded-full bg-border">
                <div className="demo-scrub-fill h-full" style={{ clipPath: `inset(0 ${100 - position}% 0 0)`, transition: glide("clip-path") }} />
              </div>
              {chapters.slice(1).map((c) => (
                <span
                  key={c.i}
                  aria-hidden
                  className="pointer-events-none absolute top-1/2 h-3 w-0.5 -translate-y-1/2 bg-foreground/70"
                  style={{ left: `${(c.i / last) * 100}%` }}
                />
              ))}
              <input
                type="range"
                min={0}
                max={last}
                value={index}
                onChange={(e) => jump(Number(e.target.value))}
                aria-label="Position in the replay"
                className="peer absolute inset-0 m-0 w-full cursor-pointer opacity-0"
              />
              <span
                aria-hidden
                className="demo-scrub-thumb pointer-events-none absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full bg-foreground peer-hover:h-5 peer-hover:w-5 peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent"
                style={{ left: `${position}%`, transition: glide("left") }}
              />
            </div>

            {chapters.length > 0 && (
              <div className="flex flex-wrap items-center gap-2 text-sm">
                <span className="text-muted">Jump to</span>
                {chapters.map((c) => (
                  <button key={c.i} type="button" className={button} onClick={() => jump(c.i)}>
                    {c.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {scenario ? <ScenarioPanel scenario={scenario} index={index} /> : <Curve demo={demo} completed={frame.completed} />}
      </div>

      {!scenario && (
        <details className="rounded-xl border border-border text-sm">
          <summary className="cursor-pointer px-4 py-3 font-medium">Results of every round as a table</summary>
          <div className="overflow-x-auto">
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
          </div>
        </details>
      )}
    </div>
  );
}
