import { ButtonLink, Card } from "@/components/ui";
import { updates } from "@/content/site";

const features = [
  {
    title: "Compatibility",
    text: "A conformance check catches clients whose feature order, units, encodings or labels differ, before they join a run.",
  },
  {
    title: "Robustness",
    text: "Slow clients, dropouts, duplicate or stale updates, and coordinator crashes that resume from stored state instead of round zero.",
  },
  {
    title: "Observability",
    text: "Event logs, latency tracking and a reproducible report at the end of every run.",
  },
];

const steps = [
  ["1", "Open", "The coordinator opens a round and each client downloads the current global model through the gateway."],
  ["2", "Train", "Each client trains on its own data. The data never leaves its machine."],
  ["3", "Report", "Clients upload only their model updates through the gateway, which stores them."],
  ["4", "Combine", "The coordinator reads the stored updates and averages them into a new global model."],
];

export default function Home() {
  const [latest, ...earlier] = updates;
  return (
    <div className="space-y-16">
      <section className="space-y-6 pt-4">
        <p className="text-sm font-medium text-accent">COMP 491 capstone project by Team FedUp</p>
        <h1 className="max-w-3xl text-4xl font-bold sm:text-5xl">
          Train one model together, without sharing the data
        </h1>
        <p className="max-w-2xl text-lg text-muted">
          A cross-silo federated learning framework: a central coordinator trains PyTorch models
          across independently operated clients, such as banks or telecom providers, without
          collecting their raw data.
        </p>
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
            <Card key={f.title}>
              <h3 className="font-semibold">{f.title}</h3>
              <p className="mt-2 text-sm text-muted">{f.text}</p>
            </Card>
          ))}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-2xl font-semibold">How a round works</h2>
        <ol className="grid gap-4 sm:grid-cols-4">
          {steps.map(([n, title, text]) => (
            <li key={n} className="space-y-2">
              <span className="grid h-8 w-8 place-items-center rounded-full bg-accent-soft text-sm font-semibold text-accent">
                {n}
              </span>
              <h3 className="font-semibold">{title}</h3>
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
        <Card>
          <div className="flex items-baseline justify-between gap-3">
            <h3 className="font-semibold">Latest: {latest.week}</h3>
            <span className="text-sm text-muted">{latest.date}</span>
          </div>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm">
            {latest.items.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </Card>
        {earlier.length > 0 && (
          <details className="rounded-xl border border-border">
            <summary className="cursor-pointer px-5 py-3 text-sm font-medium">
              Earlier weeks ({earlier.length})
            </summary>
            <div className="space-y-5 border-t border-border p-5">
              {earlier.map((u) => (
                <article key={u.week}>
                  <h3 className="text-sm font-semibold">
                    {u.week} <span className="font-normal text-muted">({u.date})</span>
                  </h3>
                  <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-muted">
                    {u.items.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </article>
              ))}
            </div>
          </details>
        )}
      </section>
    </div>
  );
}
