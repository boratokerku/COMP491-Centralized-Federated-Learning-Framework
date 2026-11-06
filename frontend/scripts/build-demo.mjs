// Builds src/content/demo-run.json for the public "how it works" page from the summary.json
// that `cfl-report` wrote for a completed run (planning/interfaces.md, section 8).
//
//   node scripts/build-demo.mjs <summary.json>
//
// Only what the animation needs leaves the report: pseudonymous client names, which clients
// reported in which round, the global metric curve and the story events. Run IDs, timestamps,
// real client identities, update details and the event log are not copied.
//
// If the run really had a dropout and a coordinator restart, those events are used. If it did
// not (local simulations cannot fail), a dropout and a restart are added and marked as
// scripted, and the page says so.
import { readFileSync, writeFileSync } from "node:fs";

const source = process.argv[2];
if (!source) {
  console.error("usage: node scripts/build-demo.mjs <summary.json>");
  process.exit(1);
}
const summary = JSON.parse(readFileSync(source, "utf8"));
if (summary.run.status !== "COMPLETED") {
  console.error(`run is ${summary.run.status}; use a completed run`);
  process.exit(1);
}

const done = summary.rounds.filter((r) => r.state === "DONE");
const ids = [...new Set(done.flatMap((r) => r.participants.map((p) => p.client_id)))].sort();
const alias = new Map(ids.map((id, i) => [id, `Client ${String.fromCharCode(65 + i)}`]));

const rounds = done.map((r, i) => ({
  round: i + 1,
  participants: r.participants.map((p) => ({ client: alias.get(p.client_id), status: p.status })),
  f1: r.global_metrics.global_f1,
  rocAuc: r.global_metrics.global_roc_auc,
  prAuc: r.global_metrics.global_pr_auc,
}));

const events = [];
const roundOf = (id) => done.findIndex((r) => r.round_id === id) + 1;
for (const inc of summary.incidents) {
  if (inc.event_type === "CLIENT_DROPOUT" && alias.has(inc.client_id) && roundOf(inc.round_id) > 0) {
    events.push({ round: roundOf(inc.round_id), type: "CLIENT_DROPOUT", client: alias.get(inc.client_id), scripted: false });
  } else if (inc.event_type === "COORDINATOR_RESUMED") {
    events.push({ round: Math.max(1, roundOf(inc.round_id)), type: "COORDINATOR_RESTART", scripted: false });
  }
}

const n = rounds.length;
if (!events.some((e) => e.type === "CLIENT_DROPOUT")) {
  const round = Math.max(2, Math.round(n * 0.35));
  const client = alias.get(ids.at(-1));
  events.push({ round, type: "CLIENT_DROPOUT", client, scripted: true });
  const r = rounds[round - 1];
  r.participants = r.participants.map((p) => (p.client === client ? { ...p, status: "DROPOUT" } : p));
}
if (!events.some((e) => e.type === "COORDINATOR_RESTART")) {
  const dropRound = events.find((e) => e.type === "CLIENT_DROPOUT").round;
  events.push({ round: Math.min(n, Math.max(Math.round(n * 0.6), dropRound + 1)), type: "COORDINATOR_RESTART", scripted: true });
}
events.sort((a, b) => a.round - b.round);

const demo = {
  task: summary.run.task,
  strategy: summary.run.strategy,
  clients: [...alias.values()],
  rounds,
  events,
  scripted: events.some((e) => e.scripted),
};
writeFileSync(new URL("../src/content/demo-run.json", import.meta.url), JSON.stringify(demo, null, 2) + "\n");
console.log(`${n} rounds, ${demo.clients.length} clients, events:`, events.map((e) => `${e.type}@${e.round}${e.scripted ? "*" : ""}`).join(", "));
