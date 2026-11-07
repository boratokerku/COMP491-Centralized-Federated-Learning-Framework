import Link from "next/link";
import type { ReactNode } from "react";

const TONES: Record<string, string> = {
  green: "bg-green-950 text-green-300",
  red: "bg-danger-soft text-danger",
  amber: "bg-warning-soft text-warning",
  blue: "bg-accent-soft text-accent",
  gray: "bg-surface text-muted",
};

const STATUS_TONE: Record<string, string> = {
  COMPLETED: "green",
  DONE: "green",
  PASSED: "green",
  UPDATE_ACCEPTED: "green",
  REGISTERED: "green",
  RUNNING: "blue",
  COLLECTING: "blue",
  OPEN: "blue",
  AGGREGATING: "blue",
  SELECTED: "amber",
  PENDING: "amber",
  APPROVED: "amber",
  FAILED: "red",
  DROPOUT: "red",
  REJECTED: "red",
  DISABLED: "red",
  ABORTED: "red",
};

export function Badge({ value }: { value: string | null | undefined }) {
  if (!value) return <span className="text-muted">-</span>;
  const tone = TONES[STATUS_TONE[value] ?? "gray"];
  return (
    <span className={`inline-block whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-medium ${tone}`}>
      {value.replaceAll("_", " ").toLowerCase()}
    </span>
  );
}

/** Title and a line that says what the page is for. */
export function PageHeader({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children?: ReactNode;
}) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-4">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold sm:text-3xl">{title}</h1>
        {description && <p className="max-w-2xl text-sm text-muted sm:text-base">{description}</p>}
      </div>
      {children}
    </header>
  );
}

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={`rounded-xl border border-border bg-surface p-5 ${className}`}>{children}</div>;
}

/** One number or word with a label: for the row of facts at the top of a page. */
export function Stat({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <div className="text-xs font-medium uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1 text-xl font-semibold">{children}</div>
      {hint && <div className="mt-1 text-xs text-muted">{hint}</div>}
    </div>
  );
}

export function Stats({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">{children}</div>;
}

/** One coordinator linked to three clients. */
export function Logo() {
  const clients = [
    { x: 6, color: "var(--danger)" },
    { x: 16, color: "var(--warning)" },
    { x: 26, color: "var(--accent)" },
  ];
  return (
    <svg viewBox="0 0 32 32" aria-hidden className="h-8 w-8 shrink-0">
      {clients.map((c) => (
        <line key={c.x} x1={16} y1={8} x2={c.x} y2={25} stroke="var(--muted)" strokeWidth={2} />
      ))}
      <circle cx={16} cy={7} r={5} fill="var(--mint)" />
      {clients.map((c) => (
        <circle key={c.x} cx={c.x} cy={25} r={4} fill={c.color} />
      ))}
    </svg>
  );
}

export function GitHubIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden className={className} fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

export function ButtonLink({
  href,
  children,
  variant = "primary",
}: {
  href: string;
  children: ReactNode;
  variant?: "primary" | "secondary";
}) {
  const style =
    variant === "primary"
      ? "bg-accent text-background hover:opacity-90"
      : "border border-border hover:bg-surface";
  return (
    <Link href={href} className={`inline-block rounded-lg px-4 py-2 text-sm font-medium transition motion-safe:hover:-translate-y-0.5 ${style}`}>
      {children}
    </Link>
  );
}

export function Table({ head, children }: { head: string[]; children: ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-border">
      <table className="w-full text-left text-sm">
        <thead className="bg-surface">
          <tr className="border-b border-border text-xs uppercase tracking-wide text-muted">
            {head.map((h) => (
              <th key={h} className="whitespace-nowrap px-3 py-2.5 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export const Row = ({ children }: { children: ReactNode }) => (
  <tr className="border-b border-border align-top last:border-0 hover:bg-surface">{children}</tr>
);
export const Cell = ({ children, mono }: { children: ReactNode; mono?: boolean }) => (
  <td className={`px-3 py-2.5 ${mono ? "break-all font-mono text-xs" : ""}`}>{children}</td>
);

export function Empty({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-xl border border-dashed border-border p-8 text-center text-sm text-muted">
      {children}
    </p>
  );
}

export function Section({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-lg font-semibold">{title}</h2>
        {hint && <p className="text-sm text-muted">{hint}</p>}
      </div>
      {children}
    </section>
  );
}

export function Facts({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="grid gap-x-8 gap-y-2.5 rounded-xl border border-border p-4 text-sm sm:grid-cols-[max-content_1fr]">
      {items.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted">{k}</dt>
          <dd className="break-all">{v}</dd>
        </div>
      ))}
    </dl>
  );
}
