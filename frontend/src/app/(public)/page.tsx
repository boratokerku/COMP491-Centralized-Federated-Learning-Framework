import { FederatedBackground } from "@/components/federated-background";
import { HeroTyping } from "@/components/hero-typing";
import type { CSSProperties } from "react";
import { ProgressTimeline } from "@/components/progress-timeline";
import { ButtonLink, Card } from "@/components/ui";
import { updates } from "@/content/site";

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

export default function Home() {
  return (
    <div className="space-y-16">
      <section className="relative isolate space-y-6 pt-4">
        <FederatedBackground />
        <p className="text-sm font-medium text-accent">COMP 491 capstone project by Team FedUp</p>
        <HeroTyping
          title="Train one model together, without sharing the data"
          text="A cross-silo federated learning framework: a central coordinator trains PyTorch models across independently operated clients, such as banks or telecom providers, without collecting their raw data."
        />
        <div className="flex flex-wrap gap-3">
          <ButtonLink href="/demo">See how it works</ButtonLink>
          <ButtonLink href="/docs" variant="secondary">
            Read the documentation
          </ButtonLink>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-2xl font-semibold">What this project adds</h2>
        <p className="max-w-3xl text-muted">
          Research code usually simulates federated learning on a single machine. A real
          deployment needs more, and that is what this project builds.
        </p>
        <div className="grid gap-4 sm:grid-cols-3">
          {features.map((f) => (
            <Card key={f.title} className={`lift-card border-t-2 ${f.tone}`}>
              <h3 className="font-semibold">{f.title}</h3>
              <p className="mt-2 text-sm text-muted">{f.text}</p>
            </Card>
          ))}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-2xl font-semibold">How a round works</h2>
        <ol className="grid gap-4 sm:grid-cols-4">
          {steps.map(([n, title, tone, text], i) => (
            <li
              key={n}
              className="round-step space-y-2"
              style={{ "--i": i, "--tone": tone, "--next": steps[i + 1]?.[2] } as CSSProperties}
            >
              <span className="round-dot grid h-8 w-8 place-items-center rounded-full bg-(--tone) text-sm font-semibold text-background">
                {n}
              </span>
              {i < steps.length - 1 && <span aria-hidden className="round-link" />}
              <h3 className="font-semibold text-(--tone)">{title}</h3>
              <p className="text-sm text-muted">{text}</p>
            </li>
          ))}
        </ol>
        <p className="text-sm text-muted">
          Two data sets are covered: tabular (Bank Marketing) and text (SMS Spam), in a local
          simulation mode and in a distributed mode over HTTPS.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-2xl font-semibold">Progress</h2>
        <ProgressTimeline updates={updates} />
      </section>
    </div>
  );
}
