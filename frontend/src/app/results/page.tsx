import type { Metadata } from "next";

export const metadata: Metadata = { title: "Results" };

const comparisons = [
  "FedAvg vs. FedProx",
  "Federated vs. centralized training",
  "Federated vs. local-only training",
];

export default function Results() {
  return (
    <div className="space-y-8">
      <h1 className="text-3xl font-bold">Results</h1>
      <p>
        Results for both modalities, Bank Marketing (tabular) and SMS Spam (text), are added
        here by hand from the run reports once the experiments are finished.
      </p>
      <section className="space-y-2">
        <h2 className="text-xl font-semibold">Planned comparisons</h2>
        <ul className="list-disc pl-6">
          {comparisons.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      </section>
      <section className="space-y-2">
        <h2 className="text-xl font-semibold">Poster and demo video</h2>
        <p className="text-muted">Will be published here when available.</p>
      </section>
    </div>
  );
}
