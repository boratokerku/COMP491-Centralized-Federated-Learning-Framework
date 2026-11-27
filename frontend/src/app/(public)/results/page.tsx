import type { Metadata } from "next";
import { LineChart } from "@/components/line-chart";
import { Card, Cell, PageHeader, Row, Table } from "@/components/ui";
import { REPO_URL } from "@/content/site";
import results from "@/content/results.json";

export const metadata: Metadata = { title: "Results" };

const blob = (path: string) => `${REPO_URL}/blob/main/${path}`;

type Case = {
  f1: number[];
  roc_auc: number[];
  pr_auc: number[];
  accuracy: number[];
  f1_per_round?: number[];
};

const METRICS = ["f1", "roc_auc", "pr_auc", "accuracy"] as const;

// Table order and names. A case that is not in results.json is left out.
const LABELS: [string, string][] = [
  ["fedavg_iid", "FedAvg, IID"],
  ["fedprox_mu0.001_iid", "FedProx μ=0.001, IID"],
  ["fedprox_mu0.01_iid", "FedProx μ=0.01, IID"],
  ["fedprox_mu0.1_iid", "FedProx μ=0.1, IID"],
  ["fedavg_label_skew", "FedAvg, label skew"],
  ["fedprox_mu0.001_label_skew", "FedProx μ=0.001, label skew"],
  ["fedprox_mu0.01_label_skew", "FedProx μ=0.01, label skew"],
  ["fedprox_mu0.1_label_skew", "FedProx μ=0.1, label skew"],
  ["centralized", "Centralized (pooled data)"],
  ["local_only", "Local only (mean of 4 clients)"],
];

const datasets: {
  key: keyof typeof results;
  title: string;
  about: string;
  findings: string[];
  note?: string;
}[] = [
  {
    key: "tabular",
    title: "Bank Marketing (tabular)",
    about: "Predicts whether a customer says yes to a bank offer. About 11% of the rows are “yes”.",
    findings: [
      "With IID clients, FedAvg and FedProx up to μ=0.01 are within 0.002 F1 of each other. μ=0.1 scores slightly lower.",
      "Clients that train alone average a lower F1 than the federated runs with IID clients (0.45 against 0.51 to 0.52).",
    ],
    note: "Label-skew results for this dataset are not shown yet: they are being checked.",
  },
  {
    key: "text",
    title: "SMS Spam (text)",
    about: "Classifies SMS messages as spam or not. About 13% of the messages are spam after removing duplicates.",
    findings: [
      "With IID clients, FedAvg reaches a higher F1 than clients that train alone (0.87 against 0.77).",
      "FedProx scores below FedAvg at both tested values of μ, with IID and with label-skewed clients.",
      "Label skew lowers F1 for every federated method. FedAvg holds up best (0.74).",
    ],
  },
];

const limits = [
  "Each result is the mean ± standard deviation of three seeds on one data split. Small differences can be noise.",
  "Training settings were chosen using the same held-out test set, so the scores are likely a little optimistic.",
  "The centralized and local-only baselines were not tuned separately. A federated score close to or above the centralized one does not show that federated training beats pooling the data.",
  "These experiments do not use the optional privacy modes.",
];

const links = [
  { label: "Experiment notes", href: blob("planning/data.md") },
  { label: "Raw result files", href: `${REPO_URL}/tree/main/results` },
];

/** One line per federated method, F1 of the global model after each round. */
function roundSeries(cases: Record<string, Case>, suffix: string) {
  return LABELS.flatMap(([key, label]) => {
    const rounds = key.endsWith(suffix) ? cases[key]?.f1_per_round : undefined;
    return rounds
      ? [{ label: label.split(",")[0], points: rounds.map((y, i) => ({ x: i + 1, y })) }]
      : [];
  });
}

export default function Results() {
  return (
    <div className="space-y-10">
      <PageHeader
        title="Results"
        description="Federated training with four clients over 20 rounds, compared with training on pooled data and training alone. Scores are measured on a test set that no client trains on."
      />

      {datasets.map((d) => {
        const cases: Record<string, Case> = results[d.key];
        const charts = [
          ["IID clients", roundSeries(cases, "_iid")],
          ["Label-skewed clients", roundSeries(cases, "_label_skew")],
        ] as const;
        return (
          <section key={d.key} className="space-y-4">
            <div className="space-y-1">
              <h2 className="text-lg font-semibold">{d.title}</h2>
              <p className="text-sm text-muted">{d.about}</p>
            </div>

            <ul className="max-w-4xl list-disc space-y-1 pl-5 text-sm text-muted">
              {d.findings.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>

            <Table head={["Setup", "F1", "ROC-AUC", "PR-AUC", "Accuracy"]}>
              {LABELS.filter(([key]) => cases[key]).map(([key, label]) => (
                <Row key={key}>
                  <Cell>{label}</Cell>
                  {METRICS.map((m) => (
                    <Cell key={m}>
                      <span className="whitespace-nowrap tabular-nums">
                        {cases[key][m][0].toFixed(3)}
                        <span className="text-muted"> ± {cases[key][m][1].toFixed(3)}</span>
                      </span>
                    </Cell>
                  ))}
                </Row>
              ))}
            </Table>
            {d.note && <p className="text-sm text-muted">{d.note}</p>}

            <div className="grid gap-4 lg:grid-cols-2">
              {charts
                .filter(([, series]) => series.length)
                .map(([title, series]) => (
                  <Card key={title}>
                    <LineChart title={`F1 per round, ${title}`} series={series} yLabel="F1" yMin={0} yMax={1} />
                  </Card>
                ))}
            </div>
          </section>
        );
      })}

      <Card>
        <h2 className="font-semibold">How to read these numbers</h2>
        <p className="mt-2 text-sm text-muted">
          IID means every client has the same share of each class. With label skew, the class shares differ strongly
          between clients (Dirichlet split, α = 0.3). The baselines use the IID client splits.
        </p>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-muted">
          {limits.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
        <ul className="mt-4 space-y-1 text-sm">
          {links.map((l) => (
            <li key={l.href}>
              <a href={l.href} className="text-accent hover:underline">
                {l.label} &rarr;
              </a>
            </li>
          ))}
        </ul>
      </Card>

      <section className="space-y-2">
        <h2 className="text-lg font-semibold">Poster and demo video</h2>
        <p className="text-sm text-muted">They will be published here when they are ready.</p>
      </section>
    </div>
  );
}
