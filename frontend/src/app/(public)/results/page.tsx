import type { Metadata } from "next";
import { Card, PageHeader } from "@/components/ui";

export const metadata: Metadata = { title: "Results" };

const comparisons = [
  ["FedAvg vs. FedProx", "Two ways of combining client updates, on both data sets."],
  ["Federated vs. centralized", "How close federated training gets to training on pooled data."],
  ["Federated vs. local only", "What each organization gains over training alone."],
];

export default function Results() {
  return (
    <div className="space-y-8">
      <PageHeader
        title="Results"
        description="Results for Bank Marketing (tabular) and SMS Spam (text) will appear here once the experiments are finished."
      />
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Planned comparisons</h2>
        <div className="grid gap-4 sm:grid-cols-3">
          {comparisons.map(([title, text]) => (
            <Card key={title}>
              <h3 className="font-semibold">{title}</h3>
              <p className="mt-1 text-sm text-muted">{text}</p>
              <p className="mt-3 text-xs font-medium uppercase tracking-wide text-muted">Coming soon</p>
            </Card>
          ))}
        </div>
      </section>
      <section className="space-y-2">
        <h2 className="text-lg font-semibold">Poster and demo video</h2>
        <p className="text-sm text-muted">They will be published here when they are ready.</p>
      </section>
    </div>
  );
}
