import assert from "node:assert/strict";
import { test } from "node:test";
import { buildFrames, type DemoRun } from "./demo-frames.ts";

const demo: DemoRun = {
  task: "text",
  strategy: "fedavg",
  clients: ["Client A", "Client B", "Client C"],
  rounds: [1, 2, 3].map((round) => ({
    round,
    participants: ["Client A", "Client B", "Client C"].map((client) => ({
      client,
      status: round === 2 && client === "Client C" ? "DROPOUT" : "UPDATE_ACCEPTED",
    })),
    f1: 0.5 + round / 10,
    rocAuc: 0.8,
    prAuc: 0.7,
  })),
  events: [
    { round: 2, type: "CLIENT_DROPOUT", client: "Client C", scripted: false },
    { round: 3, type: "COORDINATOR_RESTART", scripted: false },
  ],
  scripted: false,
};
const frames = buildFrames(demo);
const kinds = (round: number) => frames.filter((f) => f.round === round).map((f) => f.kind);

test("every round opens, collects, aggregates and finishes in that order", () => {
  assert.deepEqual(kinds(1), ["open", "report", "report", "report", "aggregate", "done"]);
});

test("a dropout shows up as its own frame and the client stays dropped for the round", () => {
  assert.deepEqual(kinds(2), ["open", "report", "report", "dropout", "aggregate", "done"]);
  const last = frames.filter((f) => f.round === 2).at(-1)!;
  assert.equal(last.clients["Client C"], "dropout");
});

test("the coordinator restart happens before aggregation of its round", () => {
  assert.deepEqual(kinds(3), ["open", "report", "report", "report", "crash", "resume", "aggregate", "done"]);
  const crash = frames.find((f) => f.kind === "crash")!;
  assert.equal(crash.coordinator, "down");
  assert.equal(frames[frames.indexOf(crash) + 1].coordinator, "collecting");
});

test("report and dropout frames name their client", () => {
  const named = frames.filter((f) => f.kind === "report" || f.kind === "dropout");
  assert.ok(named.every((f) => f.client && demo.clients.includes(f.client)));
  assert.equal(frames.find((f) => f.kind === "dropout")!.client, "Client C");
});

test("the curve only grows when a round is done", () => {
  for (const f of frames) {
    assert.equal(f.completed, f.kind === "done" ? f.round : f.round - 1);
  }
  assert.equal(frames.at(-1)!.completed, demo.rounds.length);
});

test("each round starts from fresh client states", () => {
  const second = frames.find((f) => f.round === 2 && f.kind === "open")!;
  assert.deepEqual(Object.values(second.clients), ["training", "training", "training"]);
});
