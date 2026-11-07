"use client";

import { useEffect, useState } from "react";

type Part = { text: string; className?: string };

const START_MS = 700;
const CHAR_MS = 17;
const PAUSE_MS: Record<string, number> = { ".": 260, ",": 120, ":": 160 };
const FRESH = 6; // the newest letters fade in rather than pop

/** Text that writes itself, part after part. The space it will take is held from the start, so
 * nothing below moves; screen readers and search engines get the whole text at once. */
export function TypedText({ parts, className }: { parts: Part[]; className?: string }) {
  const full = parts.map((p) => p.text).join("");
  const [count, setCount] = useState(0);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let n = 0;
    const step = () => {
      n += 1;
      setCount(n);
      if (n >= full.length) return;
      timer = setTimeout(step, CHAR_MS + (PAUSE_MS[full[n - 1]] ?? 0) + Math.random() * 14);
    };
    timer = setTimeout(() => {
      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) setCount(full.length);
      else step();
    }, START_MS);
    return () => clearTimeout(timer);
  }, [full]);

  const starts = parts.map((_, i) => parts.slice(0, i).reduce((sum, p) => sum + p.text.length, 0));
  return (
    <p className={`select-none cursor-default ${className ?? ""}`}>
      <noscript>
        <style>{".typed-rest { visibility: visible; }"}</style>
      </noscript>
      <span className="sr-only">{full}</span>
      <span aria-hidden>
        {parts.map((part, i) => {
          const from = starts[i];
          const shown = Math.max(0, Math.min(part.text.length, count - from));
          const settled = Math.max(0, shown - FRESH);
          return (
            <span key={i} className={part.className}>
              {part.text.slice(0, settled)}
              {[...part.text.slice(settled, shown)].map((ch, k) => (
                <span key={from + settled + k} className="typed-fresh">
                  {ch}
                </span>
              ))}
              <span className="typed-rest">{part.text.slice(shown)}</span>
            </span>
          );
        })}
      </span>
    </p>
  );
}
