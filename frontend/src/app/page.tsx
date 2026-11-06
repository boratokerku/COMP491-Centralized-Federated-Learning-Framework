import { updates } from "@/content/site";

export default function Home() {
  return (
    <div className="space-y-12">
      <section className="space-y-4">
        <h1 className="text-3xl font-bold">Centralized Federated Learning Framework</h1>
        <p className="text-lg text-muted">
          A cross-silo federated learning framework: a central coordinator trains PyTorch models
          across independently operated clients without collecting their raw data.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">What is cross-silo federated learning?</h2>
        <p>
          A few organizations, such as banks, hospitals or telecom providers, want to train one
          model together but cannot share their data. Each organization keeps its data on its own
          machines and trains locally. Only model updates travel to the coordinator, which
          combines them (FedAvg or FedProx) into a new global model and sends it back for the
          next round.
        </p>
        <p>
          Research code usually simulates this on a single machine. This project builds the parts
          a real deployment needs:
        </p>
        <ul className="list-disc space-y-1 pl-6">
          <li>
            <strong>Compatibility:</strong> a conformance check catches clients whose feature
            order, units, encodings or labels differ before they join a run.
          </li>
          <li>
            <strong>Robustness:</strong> slow clients, dropouts, duplicate or stale updates, and
            coordinator crashes that resume from stored state instead of round zero.
          </li>
          <li>
            <strong>Observability:</strong> event logs, latency tracking and a reproducible report
            at the end of every run.
          </li>
        </ul>
        <p>
          The framework covers two modalities: tabular (Bank Marketing) and text (SMS Spam). Runs
          work in a local simulation mode and in a distributed mode over HTTPS.
        </p>
      </section>

      <section className="space-y-5">
        <h2 className="text-xl font-semibold">Weekly progress</h2>
        {updates.map((u) => (
          <article key={u.week} className="space-y-2 border-l-2 border-border pl-4">
            <h3 className="font-medium">
              {u.week} <span className="text-sm font-normal text-muted">({u.date})</span>
            </h3>
            <ul className="list-disc space-y-1 pl-5 text-sm">
              {u.items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </article>
        ))}
      </section>
    </div>
  );
}
