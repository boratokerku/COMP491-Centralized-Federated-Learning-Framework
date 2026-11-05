import type { ReactNode } from "react";

const TONES: Record<string, string> = {
  green: "bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300",
  red: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
  amber: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  blue: "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300",
  gray: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
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
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${tone}`}>{value}</span>;
}

export function Table({ head, children }: { head: string[]; children: ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-border text-muted">
            {head.map((h) => (
              <th key={h} className="whitespace-nowrap px-2 py-2 font-medium">
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
  <tr className="border-b border-border align-top">{children}</tr>
);
export const Cell = ({ children, mono }: { children: ReactNode; mono?: boolean }) => (
  <td className={`px-2 py-2 ${mono ? "font-mono text-xs break-all" : ""}`}>{children}</td>
);

export function Empty({ children }: { children: ReactNode }) {
  return <p className="rounded border border-dashed border-border p-6 text-sm text-muted">{children}</p>;
}

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="text-lg font-semibold">{title}</h2>
      {children}
    </section>
  );
}

export function Facts({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[max-content_1fr]">
      {items.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted">{k}</dt>
          <dd className="break-all">{v}</dd>
        </div>
      ))}
    </dl>
  );
}
