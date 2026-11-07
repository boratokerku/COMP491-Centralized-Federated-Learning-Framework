import { HeroNetwork } from "@/components/hero-network";
import { SectionTitle } from "@/components/section-title";
import type { CSSProperties } from "react";
import { ProgressTimeline } from "@/components/progress-timeline";
import { TypedText } from "@/components/typed-text";
import { ButtonLink, Card, GitHubIcon } from "@/components/ui";
import { REPO_URL, updates } from "@/content/site";

const features = [
  {
    title: "Compatibility",
    tone: "border-t-danger text-danger",
    text: "A conformance check catches clients whose feature order, units, encodings or labels differ, before they join a run.",
  },
  {
    title: "Robustness",
    tone: "border-t-warning text-warning",
    text: "Slow clients, dropouts, duplicate or stale updates, and coordinator crashes that resume from stored state instead of round zero.",
  },
  {
    title: "Observability",
    tone: "border-t-accent text-accent",
    text: "Event logs, latency tracking and a reproducible report at the end of every run.",
  },
];

const steps = [
  ["1", "Open", "var(--danger)", "The coordinator opens a round and each client downloads the current global model through the gateway."],
  ["2", "Train", "var(--warning)", "Each client trains on its own data. The data never leaves its machine."],
  ["3", "Report", "var(--accent)", "Clients upload only their model updates through the gateway, which stores them."],
  ["4", "Combine", "var(--mint)", "The coordinator reads the stored updates and averages them into a new global model."],
];

const RING = "SCROLL TO EXPLORE · SCROLL TO EXPLORE · ";

export default function Home() {
  return (
    <>
      {/* The landing screen: fills the window below the menu. "Enter" scrolls into the site. */}
      <section className="relative isolate -mt-12 grid min-h-[calc(100svh-3.5rem)] items-center gap-8 pt-10 pb-28 lg:grid-cols-[1.05fr_1fr] lg:gap-4 lg:pt-0">
        <div aria-hidden className="hero-glow" />

        <div className="hero-in space-y-7">
          <p className="text-xs font-medium uppercase tracking-[0.3em] text-accent">
            COMP 491 capstone · Koç University
          </p>
          <h1>
            <span className="block text-lg font-medium uppercase tracking-[0.4em] text-foreground/80 sm:text-2xl">
              We are
            </span>
            <span className="hero-title mt-1 block">
              Fed<span className="hero-title-up">Up</span>
            </span>
          </h1>
          <TypedText
            className="max-w-xl text-lg text-muted sm:text-xl"
            parts={[
              { text: "Train one model together, without sharing the data. ", className: "text-foreground" },
              {
                text: "A cross-silo federated learning framework: a central coordinator trains PyTorch models across independently operated clients, such as banks or telecom providers, without collecting their raw data.",
              },
            ]}
          />
          <div className="flex flex-wrap items-center gap-4">
            <a href="#explore" className="enter-pill">
              Enter the site
              <span aria-hidden className="enter-pill-knob">
                <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2}>
                  <path d="M3 8h10M9 4l4 4-4 4" />
                </svg>
              </span>
            </a>
            <ButtonLink href="/demo" variant="secondary">
              See how it works
            </ButtonLink>
          </div>
        </div>

        <div className="hero-in hero-in-late">
          <HeroNetwork />
        </div>

        <div className="absolute inset-x-0 bottom-6 flex items-center justify-between text-sm text-muted">
          <a href={REPO_URL} className="inline-flex items-center gap-2 transition-colors hover:text-foreground">
            <GitHubIcon className="h-4 w-4" />
            <span className="hidden sm:inline">Source code</span>
          </a>
          <a href="#explore" aria-label="Scroll to explore" className="scroll-ring">
            <span aria-hidden className="scroll-ring-text">
              {[...RING].map((ch, i) => (
                <span key={i} style={{ transform: `rotate(${(i * 360) / RING.length}deg)` }}>
                  {ch}
                </span>
              ))}
            </span>
            <svg viewBox="0 0 16 16" aria-hidden className="absolute h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2}>
              <path d="M8 3v10M4 9l4 4 4-4" />
            </svg>
          </a>
          <span className="hidden sm:inline">Bank Marketing · SMS Spam</span>
          <span className="sm:hidden" />
        </div>
      </section>

      <div id="explore" className="scroll-mt-16 space-y-28 pt-8 pb-8">
        <section className="space-y-8">
          <SectionTitle outline="About" eyebrow="What this project adds">
            Research code simulates federated learning on one machine. A real deployment needs more.
          </SectionTitle>
          <div className="grid gap-5 md:grid-cols-3">
            {features.map((f) => (
              <Card key={f.title} className={`lift-card border-t-2 p-6 ${f.tone}`}>
                <h3 className="text-lg font-semibold">{f.title}</h3>
                <p className="mt-3 text-muted">{f.text}</p>
              </Card>
            ))}
          </div>
        </section>

        <section className="space-y-10">
          <SectionTitle outline="Process" eyebrow="How a round works">
            Four steps, repeated every round. The raw data never leaves a client.
          </SectionTitle>
          <ol className="grid gap-8 sm:grid-cols-4">
            {steps.map(([n, title, tone, text], i) => (
              <li
                key={n}
                className="round-step space-y-3"
                style={{ "--i": i, "--tone": tone, "--next": steps[i + 1]?.[2] } as CSSProperties}
              >
                <span className="round-dot grid h-8 w-8 place-items-center rounded-full bg-(--tone) text-sm font-semibold text-background">
                  {n}
                </span>
                {i < steps.length - 1 && <span aria-hidden className="round-link" />}
                <h3 className="text-lg font-semibold text-(--tone)">{title}</h3>
                <p className="text-muted">{text}</p>
              </li>
            ))}
          </ol>
          <p className="text-muted">
            Two data sets are covered: tabular (Bank Marketing) and text (SMS Spam), in a local
            simulation mode and in a distributed mode over HTTPS.
          </p>
        </section>

        <section className="grid gap-10 lg:grid-cols-[1fr_2fr]">
          <div className="lg:sticky lg:top-24 lg:self-start">
            <SectionTitle outline="Weekly" eyebrow="Progress" from="left">
              What we built, week by week.
            </SectionTitle>
          </div>
          <ProgressTimeline updates={updates} />
        </section>
      </div>
    </>
  );
}
