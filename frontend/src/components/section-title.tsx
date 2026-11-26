import type { ReactNode } from "react";

/** A section heading with a large outlined word behind it. */
export function SectionTitle({
  outline,
  eyebrow,
  children,
  from = "right",
}: {
  outline: string;
  /** Which edge the outlined word is pinned to; "left" for narrow columns. */
  from?: "left" | "right";
  eyebrow: string;
  children: ReactNode;
}) {
  return (
    <div className="relative isolate pt-10">
      <span aria-hidden className="outline-word" style={from === "left" ? { left: 0, right: "auto" } : undefined}>
        {outline}
      </span>
      <p className="text-xs font-medium uppercase tracking-[0.25em] text-muted">{eyebrow}</p>
      <h2 className="mt-3 max-w-2xl text-3xl font-semibold sm:text-4xl">{children}</h2>
    </div>
  );
}
