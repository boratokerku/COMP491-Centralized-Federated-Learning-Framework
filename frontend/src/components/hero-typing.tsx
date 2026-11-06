"use client";

import { useEffect, useState } from "react";

const START_DELAY_MS = 300;
const TITLE_CHAR_MS = 45;
const PAUSE_BETWEEN_MS = 350;
const TEXT_CHAR_MS = 14;

// The part still to come stays in the line as hidden text, so words never jump between lines
// and the page below does not move. Screen readers and search engines get the whole text at once.
function Typed({ text, shown, active }: { text: string; shown: number; active: boolean }) {
  return (
    <>
      <span className="sr-only">{text}</span>
      <span aria-hidden>
        {text.slice(0, shown)}
        {active && <span className="typing-caret" />}
        <span className="typing-rest">{text.slice(shown)}</span>
      </span>
    </>
  );
}

/** The home page title and intro, typed one after the other. */
export function HeroTyping({ title, text }: { title: string; text: string }) {
  const total = title.length + text.length;
  const [count, setCount] = useState(0);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let next = 0;
    const step = () => {
      next += 1;
      setCount(next);
      if (next >= total) return;
      if (next === title.length) timer = setTimeout(step, PAUSE_BETWEEN_MS);
      else timer = setTimeout(step, next < title.length ? TITLE_CHAR_MS : TEXT_CHAR_MS);
    };
    timer = setTimeout(
      () => {
        if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) setCount(total);
        else step();
      },
      START_DELAY_MS,
    );
    return () => clearTimeout(timer);
  }, [title, total]);

  return (
    <>
      <noscript>
        <style>{".typing-rest { visibility: visible; }"}</style>
      </noscript>
      <h1 className="max-w-3xl text-4xl font-bold sm:text-5xl">
        <Typed text={title} shown={Math.min(count, title.length)} active={count < title.length} />
      </h1>
      <p className="max-w-2xl text-lg text-muted">
        <Typed
          text={text}
          shown={Math.max(0, count - title.length)}
          active={count >= title.length && count < total}
        />
      </p>
    </>
  );
}
