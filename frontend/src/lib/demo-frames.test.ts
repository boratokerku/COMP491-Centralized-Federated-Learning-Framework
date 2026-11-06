import assert from "node:assert/strict";
import { test } from "node:test";
import { buildFrames, buildScenarios, COORDINATOR, DATABASE, GATEWAY, type DemoRun } from "./demo-frames.ts";

const names = ["Client A", "Client B", "Client C"];
const demo: DemoRun = {
  task: "text",
  strategy: "fedavg",
  clients: names,
  rounds: [1, 2, 3, 4].map((round) => ({
    round,
    participants: names.map((client) => ({
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

test("the first round shows every step and each client's report on its own", () => {
  assert.deepEqual(kinds(1), ["open", "train", "report", "report", "report", "aggregate", "done"]);
});

test("the global model reaches every client when a round opens", () => {
  const opening = frames.find((f) => f.kind === "open")!;
  assert.deepEqual(opening.packets.map((p) => p.to), names);
  assert.ok(opening.packets.every((p) => p.from === DATABASE && p.kind === "model" && p.label === "v0"));
});

test("a dropout waits for the deadline and the client stays dropped for the round", () => {
  assert.deepEqual(kinds(2), ["open", "train", "report", "wait", "dropout", "aggregate", "done"]);
  const last = frames.filter((f) => f.round === 2).at(-1)!;
  assert.equal(last.clients["Client C"], "dropout");
  assert.equal(frames.find((f) => f.kind === "dropout")!.client, "Client C");
});

test("the coordinator restart happens before aggregation of its round", () => {
  assert.deepEqual(kinds(3), ["open", "train", "report", "crash", "resume", "aggregate", "done"]);
  const crash = frames.find((f) => f.kind === "crash")!;
  assert.equal(crash.coordinator, "down");
  const resume = frames[frames.indexOf(crash) + 1];
  assert.deepEqual(resume.packets, [{ from: DATABASE, to: COORDINATOR, kind: "state" }]);
});

test("a normal round runs faster than the narrated first round, under one caption", () => {
  const time = (round: number) => frames.filter((f) => f.round === round).reduce((t, f) => t + f.duration, 0);
  assert.ok(time(4) < time(1) / 2);
  assert.equal(new Set(frames.filter((f) => f.round === 4).map((f) => f.caption)).size, 1);
});

test("a client that dropped out is explained when it returns", () => {
  const rounds = demo.rounds.slice(0, 3);
  const noRestart = buildFrames({ ...demo, rounds, events: [demo.events[0]] });
  assert.match(noRestart.find((f) => f.round === 3)!.caption, /Client C is back/);
  assert.doesNotMatch(frames.find((f) => f.round === 4)!.caption, /back/);
});

test("an update counts as sent only after the frame that carries it", () => {
  const i = frames.findIndex((f) => f.kind === "report");
  assert.equal(frames[i].clients["Client A"], "sending");
  assert.equal(frames[i].received, 0);
  assert.equal(frames[i + 1].clients["Client A"], "sent");
  assert.equal(frames[i + 1].received, 1);
});

test("the curve only grows when a round is done", () => {
  for (const f of frames) {
    assert.equal(f.completed, f.kind === "done" ? f.round : f.round - 1);
  }
  assert.equal(frames.at(-1)!.completed, demo.rounds.length);
});

test("each round starts from fresh client states", () => {
  const second = frames.find((f) => f.round === 2 && f.kind === "open")!;
  assert.deepEqual(Object.values(second.clients), ["idle", "idle", "idle"]);
});

test("jump targets mark the normal round and both failures", () => {
  assert.deepEqual(frames.flatMap((f) => f.chapter ?? []), ["A normal round", "Client dropout", "Coordinator restart"]);
});

test("every report frame moves an update for each client it names", () => {
  for (const f of frames.filter((x) => x.kind === "report")) {
    assert.ok(f.packets.every((p) => p.kind === "update" && p.to === DATABASE));
    if (f.client) assert.equal(f.packets[0].from, f.client);
  }
});

const scenarios = buildScenarios();
const scenario = (id: string) => scenarios.find((s) => s.id === id)!;
const events = (id: string) => scenario(id).frames.flatMap((f) => f.event?.name ?? []);
const details = (id: string) => scenario(id).frames.flatMap((f) => f.event?.detail ?? []);

test("scenario ids are unique and every scenario is watchable", () => {
  assert.equal(new Set(scenarios.map((s) => s.id)).size, scenarios.length);
  for (const s of scenarios) {
    assert.ok(s.frames.length >= 4);
    assert.ok(s.frames.every((f) => f.caption && f.duration > 0));
  }
});

test("a dropped client is selected again in the next round", () => {
  const frames = scenario("dropout").frames;
  assert.ok(events("dropout").includes("CLIENT_DROPOUT"));
  assert.deepEqual(frames.at(-1)!.packets.map((p) => p.to), ["Client A", "Client B", "Client C", "Client D"]);
  assert.equal(frames.at(-1)!.model, 4);
});

test("updates that arrive during a crash wait in the database", () => {
  const frames = scenario("crash").frames;
  const outage = frames.find((f) => f.coordinator === "down" && f.packets.length > 0)!;
  assert.ok(outage.packets.every((p) => p.to === DATABASE));
  assert.ok(events("crash").includes("COORDINATOR_RESUMED"));
  assert.equal(frames.at(-1)!.received, 4);
});

test("a round below the quorum fails and leaves the model unchanged", () => {
  const frames = scenario("quorum").frames;
  const failed = frames.find((f) => f.kind === "failed")!;
  assert.ok(failed.received < failed.quorum!);
  assert.equal(failed.model, 3);
  assert.equal(frames.at(-1)!.model, 3);
  assert.ok(details("quorum").includes("quorum_not_met"));
});

test("a late update is refused without changing the new model", () => {
  const refused = scenario("late").frames.find((f) => f.kind === "refused")!;
  assert.equal(refused.packets[0].fate, "refused");
  assert.equal(refused.model, 4);
  assert.ok(details("late").includes("stale_round"));
});

test("a resent update counts once", () => {
  const frames = scenario("retry").frames;
  assert.equal(frames.at(-1)!.received, 4);
  assert.ok(events("retry").includes("UPDATE_RETRY_ACKNOWLEDGED"));
});

test("a broken update never reaches the average", () => {
  const frames = scenario("broken").frames;
  const aggregate = frames.find((f) => f.kind === "aggregate")!;
  assert.equal(aggregate.received, 3);
  assert.equal(aggregate.clients["Client D"], "rejected");
  assert.ok(details("broken").includes("sanity_check_failed"));
});

test("the broken update is excluded after aggregation starts, as the system logs it", () => {
  const log = events("broken");
  assert.ok(log.indexOf("ROUND_AGGREGATING") < log.indexOf("UPDATE_EXCLUDED"));
  assert.equal(log.filter((e) => e === "ROUND_AGGREGATING").length, 1);
});

test("clients never talk to the coordinator directly", () => {
  const all = [...frames, ...scenarios.flatMap((s) => s.frames)];
  for (const { from, to } of all.flatMap((f) => f.packets)) {
    if (from === COORDINATOR) assert.equal(to, DATABASE);
    if (to === COORDINATOR) assert.equal(from, DATABASE);
  }
});

test("a client that fails conformance is refused by the gateway and gets no model", () => {
  const opening = scenario("conformance").frames.find((f) => f.kind === "open")!;
  assert.ok(opening.packets.every((p) => p.to !== "Client D"));
  const reports = scenario("conformance").frames[0].packets;
  assert.ok(reports.every((p) => p.to === GATEWAY));
  assert.equal(opening.clients["Client D"], "blocked");
  assert.equal(opening.expected, 3);
});
