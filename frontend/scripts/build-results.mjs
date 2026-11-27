// Builds src/content/results.json for the public Results page from the comparison runs in
// ../results/comparison. The website image only contains frontend/, so the numbers are copied
// in here. Run from frontend/ after the experiments change: node scripts/build-results.mjs
import { readFileSync, writeFileSync } from "node:fs";

const root = new URL("../../results/comparison/", import.meta.url);
const read = (path) => readFileSync(new URL(path, root), "utf8");

// Cases left out of the page.
// tabular label skew: these runs used max_num_samples 10000, but one skewed client has more
// training rows than that, so its updates were over the cap. Remove after a re-run.
const SKIP = { tabular: (name) => name.endsWith("_label_skew") };

const METRICS = ["f1", "roc_auc", "pr_auc", "accuracy"];

function build(task) {
  const skip = SKIP[task] ?? (() => false);
  const [header, ...lines] = read(`${task}/summary.csv`).trim().split(/\r?\n/);
  const columns = header.split(",");
  const { runs } = JSON.parse(read(`${task}/results.json`));
  const cases = {};

  for (const line of lines) {
    const row = Object.fromEntries(line.split(",").map((v, i) => [columns[i], v]));
    // local_only_client_test is scored on other test data, so it is not comparable.
    if (row.case === "local_only_client_test" || skip(row.case)) continue;

    const entry = {};
    for (const m of METRICS) entry[m] = [Number(row[`${m}_mean`]), Number(row[`${m}_std`])];

    // Global F1 per round, averaged over the seeds (federated runs only).
    const seeds = runs[row.case];
    if (Array.isArray(seeds) && seeds[0].rounds) {
      entry.f1_per_round = seeds[0].rounds.map((_, i) =>
        Number((seeds.reduce((sum, s) => sum + s.rounds[i].f1, 0) / seeds.length).toFixed(4)),
      );
    }
    cases[row.case] = entry;
  }
  return cases;
}

const out = { tabular: build("tabular"), text: build("text") };
writeFileSync(new URL("../src/content/results.json", import.meta.url), JSON.stringify(out, null, 1) + "\n");
console.log(Object.entries(out).map(([t, c]) => `${t}: ${Object.keys(c).join(", ")}`).join("\n"));
