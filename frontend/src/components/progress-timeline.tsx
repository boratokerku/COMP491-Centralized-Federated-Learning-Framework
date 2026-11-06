"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";

type Week = { week: string; date: string; items: string[] };

const TONES = ["var(--accent)", "var(--danger)", "var(--warning)", "var(--mint)"];
const toneOf = (i: number) => TONES[i % TONES.length];

// Colours go by position from the oldest week, so a new week never recolours the earlier ones.
// How far down the window the line has to reach before a week lights up.
const LINE_AT = 0.65;

/** The weekly notes as rounds on a line that fills while scrolling. Takes them newest first. */
export function ProgressTimeline({ updates }: { updates: Week[] }) {
  const weeks = [...updates].reverse();
  const last = weeks.length - 1;
  const [open, setOpen] = useState<number[]>([last]);
  const list = useRef<HTMLOListElement>(null);

  // The line is set on the element, outside React, so scrolling never causes a render.
  useEffect(() => {
    const root = list.current;
    if (!root) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const dots = [...root.querySelectorAll<HTMLElement>(".timeline-dot")];
    const middle = (el: HTMLElement) => {
      const box = el.getBoundingClientRect();
      return box.top + box.height / 2;
    };
    let frame = 0;

    const measure = () => {
      frame = 0;
      const start = middle(dots[0]);
      const end = middle(dots[dots.length - 1]);
      const atBottom = window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2;
      const reach = reduced || atBottom ? Infinity : window.innerHeight * LINE_AT;
      const stops = dots.map((dot, i) => `${toneOf(i)} ${middle(dot) - start}px`);
      root.style.setProperty("--rail", `${end - start}px`);
      root.style.setProperty("--rail-colors", `linear-gradient(180deg, ${stops.join(", ")})`);
      root.style.setProperty("--fill", `${Math.min(end - start, Math.max(0, reach - start))}px`);
      for (const dot of dots) {
        if (middle(dot) <= reach) dot.closest("li")?.setAttribute("data-lit", "");
      }
    };
    const request = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };

    root.setAttribute("data-ready", "");
    measure();
    const sizes = new ResizeObserver(request);
    sizes.observe(root);
    window.addEventListener("scroll", request, { passive: true });
    window.addEventListener("resize", request);
    return () => {
      cancelAnimationFrame(frame);
      sizes.disconnect();
      window.removeEventListener("scroll", request);
      window.removeEventListener("resize", request);
    };
  }, []);

  const toggle = (i: number) =>
    setOpen((now) => (now.includes(i) ? now.filter((n) => n !== i) : [...now, i]));

  return (
    <ol ref={list} className="timeline space-y-4">
      {weeks.map((w, i) => {
        const isOpen = open.includes(i);
        return (
          <li
            key={w.week}
            className={`relative pl-12 ${i === last ? "timeline-latest" : ""}`}
            style={{ "--tone": toneOf(i) } as CSSProperties}
          >
            <span aria-hidden className="timeline-dot absolute top-2 left-0 grid h-8 w-8 place-items-center rounded-full bg-(--tone) text-xs font-semibold text-background">
              v{i + 1}
            </span>
            <div className="timeline-body rounded-xl border border-border bg-surface">
              <button
                type="button"
                aria-expanded={isOpen}
                onClick={() => toggle(i)}
                className="flex w-full flex-wrap items-baseline gap-x-3 gap-y-1 px-5 py-3 text-left"
              >
                <h3 className="font-semibold text-(--tone)">{w.week}</h3>
                {i === last && <span className="text-xs font-medium text-muted">Latest</span>}
                <span className="ml-auto text-sm text-muted">{w.date}</span>
                <span aria-hidden className={`text-muted transition-transform ${isOpen ? "rotate-90" : ""}`}>
                  &rsaquo;
                </span>
              </button>
              <div className="timeline-panel" data-open={isOpen || undefined} inert={!isOpen}>
                <ul className="min-h-0 list-disc space-y-1.5 overflow-hidden pr-5 pl-10 text-sm">
                  {w.items.map((item, n) => (
                    <li key={item} className="timeline-bullet" style={{ "--n": n } as CSSProperties}>
                      {item}
                    </li>
                  ))}
                  <li aria-hidden className="h-3 list-none" />
                </ul>
              </div>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
