// The replays on the public "how it works" page, as lists of frames: one for a finished run and
// one for each failure scenario. Kept free of React and of imports so it can be tested with
// `npm test`.
//
// Scenario steps follow what the backend does (coordinator.py, intake.py, aggregator.py), and
// event and reason names are the ones it logs. Keep them in step when those rules change.

export type DemoRun = {
  task: string;
  strategy: string;
  clients: string[];
  rounds: {
    round: number;
    participants: { client: string; status: string }[];
    f1: number;
    rocAuc: number;
    prAuc: number;
  }[];
  events: { round: number; type: "CLIENT_DROPOUT" | "COORDINATOR_RESTART"; client?: string; scripted: boolean }[];
  scripted: boolean;
};

export type ClientState =
  | "idle"
  | "checking"
  | "blocked"
  | "training"
  | "sending"
  | "sent"
  | "dropout"
  | "late"
  | "rejected";
export type CoordinatorState =
  | "waiting"
  | "checking"
  | "sending"
  | "collecting"
  | "down"
  | "restarting"
  | "aggregating"
  | "done"
  | "failed";

export const COORDINATOR = "coordinator";
export const DATABASE = "database";

export type Packet = {
  /** COORDINATOR, DATABASE or a client name. */
  from: string;
  to: string;
  kind: "model" | "update" | "broken" | "state" | "ack" | "report";
  label?: string;
  /** "refused": stopped at the door. "lost": never arrives. */
  fate?: "refused" | "lost";
};

export type Frame = {
  kind:
    | "conformance"
    | "blocked"
    | "open"
    | "train"
    | "report"
    | "wait"
    | "dropout"
    | "crash"
    | "resume"
    | "refused"
    | "aggregate"
    | "done"
    | "failed";
  round: number;
  /** The client this frame is about, when there is one. */
  client?: string;
  clients: Record<string, ClientState>;
  coordinator: CoordinatorState;
  /** Version of the global model the coordinator holds. */
  model: number;
  received: number;
  expected: number;
  /** Minimum updates for a valid round, shown when a scenario depends on it. */
  quorum: number | null;
  /** Share of the deadline left at the end of this frame; null when no deadline runs. */
  deadline: number | null;
  /** The database is being written or read. */
  database: boolean;
  packets: Packet[];
  /** Rounds whose global model is already on the curve. */
  completed: number;
  caption: string;
  /** What the system logs at this point. */
  event?: { name: string; detail?: string };
  /** Milliseconds at 1x. */
  duration: number;
  /** Set on the first frame of a part of the replay worth jumping to. */
  chapter?: string;
};

type Extra = Partial<Pick<Frame, "client" | "packets" | "database" | "event" | "chapter">>;

// A mutable snapshot of the system, plus the frames taken of it so far.
function recorder(clients: string[], first: { model: number; quorum?: number }) {
  const frames: Frame[] = [];
  const s = {
    round: first.model + 1,
    clients: Object.fromEntries(clients.map((c) => [c, "idle" as ClientState])),
    coordinator: "waiting" as CoordinatorState,
    model: first.model,
    received: 0,
    expected: clients.length,
    quorum: first.quorum ?? null,
    deadline: null as number | null,
    completed: 0,
  };
  const push = (kind: Frame["kind"], duration: number, caption: string, extra: Extra = {}) =>
    frames.push({ ...s, clients: { ...s.clients }, kind, duration, caption, packets: [], database: false, ...extra });
  return { s, frames, push };
}

type Recorder = ReturnType<typeof recorder>;

function open(r: Recorder, names: string[], duration: number, caption: string, extra: Extra = {}) {
  const { s } = r;
  for (const c of names) s.clients[c] = "idle";
  s.coordinator = "sending";
  s.expected = names.length;
  s.received = 0;
  s.deadline = 1;
  r.push("open", duration, caption, {
    packets: names.map((to) => ({ from: COORDINATOR, to, kind: "model", label: `v${s.model}` })),
    event: { name: "ROUND_OPENED" },
    ...extra,
  });
}

function train(r: Recorder, names: string[], duration: number, caption: string, deadline: number) {
  for (const c of names) r.s.clients[c] = "training";
  r.s.coordinator = "collecting";
  r.s.deadline = deadline;
  r.push("train", duration, caption);
}

// Clients show as sending during the frame and as sent from the next one, once the update landed.
function report(r: Recorder, names: string[], duration: number, caption: string, deadline: number, extra: Extra = {}) {
  const { s } = r;
  for (const c of names) s.clients[c] = "sending";
  s.deadline = deadline;
  r.push("report", duration, caption, {
    client: names.length === 1 ? names[0] : undefined,
    packets: names.map((from) => ({ from, to: COORDINATOR, kind: "update" })),
    event: { name: "UPDATE_ACCEPTED", detail: names.join(", ") },
    ...extra,
  });
  for (const c of names) s.clients[c] = "sent";
  s.received += names.length;
}

function aggregate(r: Recorder, duration: number, caption: string) {
  r.s.coordinator = "aggregating";
  r.s.deadline = null;
  r.push("aggregate", duration, caption, {
    event: { name: "ROUND_AGGREGATING", detail: `${r.s.received} updates` },
  });
  r.s.model += 1;
  r.s.coordinator = "done";
}

const list = (names: string[]) =>
  names.length < 2 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;

export function buildFrames(demo: DemoRun): Frame[] {
  const r = recorder(demo.clients, { model: 0 });
  const { s, push } = r;

  for (const rd of demo.rounds) {
    s.round = rd.round;
    s.model = rd.round - 1;
    const sent = rd.participants.filter((p) => p.status === "UPDATE_ACCEPTED").map((p) => p.client);
    const missing = rd.participants.filter((p) => p.status !== "UPDATE_ACCEPTED").map((p) => p.client);
    const crash = demo.events.some((e) => e.round === rd.round && e.type === "COORDINATOR_RESTART");
    const last = demo.rounds[rd.round - 2];
    const back = last?.participants.filter((p) => p.status !== "UPDATE_ACCEPTED").map((p) => p.client) ?? [];
    const done = () => {
      s.completed = rd.round;
      push("done", 1400, `Model v${s.model} is evaluated: F1 ${rd.f1.toFixed(2)}, ROC-AUC ${rd.rocAuc.toFixed(2)}.`, {
        event: { name: "ROUND_DONE", detail: `v${s.model}` },
      });
    };

    if (rd.round === 1) {
      open(r, demo.clients, 1700, "Round 1: the coordinator sends the current global model to every client.", {
        chapter: "A normal round",
      });
      train(r, demo.clients, 1800, "Each client trains on its own data. The data never leaves its machine.", 0.7);
      for (const [i, c] of sent.entries()) {
        report(r, [c], 850, `${c} sends only its model update. Its data stays on its own machine.`, 0.6 - i * 0.1);
      }
      aggregate(r, 1500, "The coordinator combines the received updates into a new global model (FedAvg).");
      done();
    } else if (missing.length > 0 || crash) {
      const chapter = missing.length > 0 ? "Client dropout" : "Coordinator restart";
      open(r, demo.clients, 800, `Round ${rd.round}: the clients receive model v${s.model}.`, { chapter });
      train(r, demo.clients, 800, `Round ${rd.round}: the clients train on their own data.`, 0.6);
      report(r, sent, 900, `${list(sent)} send${sent.length === 1 ? "s" : ""} an update.`, 0.45);
      if (missing.length > 0) {
        s.deadline = 0;
        push("wait", 1500, `The coordinator waits for ${list(missing)} until the deadline runs out.`);
        for (const c of missing) s.clients[c] = "dropout";
        push("dropout", 1800, `${list(missing)} did not answer before the deadline. The round continues without it.`, {
          client: missing.length === 1 ? missing[0] : undefined,
          event: { name: "CLIENT_DROPOUT", detail: "deadline_no_update" },
        });
      }
      if (crash) {
        s.coordinator = "down";
        push("crash", 2000, "The coordinator process stops. The state of the round is kept in the database.", {
          database: true,
        });
        s.coordinator = "restarting";
        push("resume", 2200, "The coordinator restarts, reads the round from the database and carries on. No update is lost.", {
          database: true,
          packets: [{ from: DATABASE, to: COORDINATOR, kind: "state" }],
          event: { name: "COORDINATOR_RESUMED" },
        });
      }
      aggregate(r, 1300, `The coordinator combines the ${s.received} received updates into a new global model (FedAvg).`);
      done();
    } else {
      // Nothing new happens in a normal round, so it runs quickly under one caption.
      const returned = back.length > 0 ? ` ${list(back)} is back: a dropout costs only one round.` : "";
      const caption = `Round ${rd.round}: a normal round. All clients train and report, and the model moves to v${rd.round}.${returned}`;
      const pace = back.length > 0 ? 1.6 : 1;
      open(r, demo.clients, 600 * pace, caption);
      train(r, demo.clients, 500 * pace, caption, 0.6);
      report(r, sent, 700 * pace, caption, 0.4);
      aggregate(r, 450 * pace, caption);
      s.completed = rd.round;
      push("done", 700 * pace, caption, { event: { name: "ROUND_DONE", detail: `v${s.model}` } });
    }
  }
  return r.frames;
}

export type Scenario = {
  id: string;
  title: string;
  /** What goes wrong, in one line. */
  problem: string;
  /** How the framework ends up. */
  outcome: string;
  frames: Frame[];
};

const A = "Client A";
const B = "Client B";
const C = "Client C";
const D = "Client D";
const ALL = [A, B, C, D];
const QUORUM = 2;

function opening(r: Recorder, names = ALL) {
  open(r, names, 1200, `Round ${r.s.round}: the coordinator sends model v${r.s.model} to the clients and starts the deadline.`);
  train(r, names, 1200, "The clients train on their own data.", 0.75);
}

function dropoutScenario(): Scenario {
  const r = recorder(ALL, { model: 3, quorum: QUORUM });
  const { s, push } = r;
  opening(r);
  report(r, [A, B, C], 1100, "Client A, Client B and Client C send their updates.", 0.5);
  s.deadline = 0;
  push("wait", 2000, "Client D has gone silent. The coordinator does not wait forever: the round has a deadline.");
  s.clients[D] = "dropout";
  push("dropout", 2200, "The deadline passes. Client D is marked as dropped out for this round.", {
    client: D,
    event: { name: "CLIENT_DROPOUT", detail: "deadline_no_update" },
  });
  aggregate(r, 1700, `3 updates are above the minimum of ${QUORUM}, so FedAvg runs on those three.`);
  push("done", 1500, "Model v4 is ready. Training moved on without Client D.", {
    event: { name: "ROUND_DONE", detail: "v4" },
  });
  s.round += 1;
  open(r, ALL, 1800, "Round 5 opens and Client D is selected again. If it is back, it simply rejoins with model v4.");
  return {
    id: "dropout",
    title: "A client disappears",
    problem: "A client goes silent in the middle of a round.",
    outcome: "The round closes at the deadline without it. The client can rejoin in the next round.",
    frames: r.frames,
  };
}

function crashScenario(): Scenario {
  const r = recorder(ALL, { model: 3, quorum: QUORUM });
  const { s, push } = r;
  opening(r);
  report(r, [A, B], 1100, "Client A and Client B send their updates. They are written to the database.", 0.55, {
    database: true,
  });
  s.coordinator = "down";
  push("crash", 2000, "The coordinator process crashes in the middle of the round.");
  for (const c of [C, D]) s.clients[c] = "sending";
  s.deadline = 0.3;
  push("report", 2200, "The gateway is a separate service, so it keeps accepting updates and stores them while the coordinator is down.", {
    database: true,
    packets: [C, D].map((from) => ({ from, to: DATABASE, kind: "update" })),
    event: { name: "UPDATE_ACCEPTED", detail: "Client C, Client D" },
  });
  for (const c of [C, D]) s.clients[c] = "sent";
  s.received = 4;
  s.coordinator = "restarting";
  push("resume", 2500, "The coordinator restarts, checks the checksum of the stored model and reads the round back.", {
    database: true,
    packets: [{ from: DATABASE, to: COORDINATOR, kind: "state" }],
    event: { name: "COORDINATOR_RESUMED" },
  });
  s.coordinator = "collecting";
  push("resume", 1500, "All 4 updates are there, including the two that arrived during the outage. The run continues from round 4, not from zero.");
  aggregate(r, 1500, "The round finishes as if nothing had happened.");
  push("done", 1500, "Model v4 is ready. No client had to train again.", { event: { name: "ROUND_DONE", detail: "v4" } });
  return {
    id: "crash",
    title: "The coordinator crashes",
    problem: "The coordinator process stops while a round is running.",
    outcome: "It restarts from the stored state. No work is lost and no client trains twice.",
    frames: r.frames,
  };
}

function quorumScenario(): Scenario {
  const r = recorder(ALL, { model: 3, quorum: QUORUM });
  const { s, push } = r;
  opening(r);
  report(r, [A], 1200, "Client A sends its update. The other three stay silent, for example during a network outage.", 0.45);
  s.deadline = 0;
  push("wait", 2000, `A valid round needs at least ${QUORUM} updates, and only 1 has arrived.`);
  for (const c of [B, C, D]) s.clients[c] = "dropout";
  push("dropout", 1800, "The deadline passes. Client B, Client C and Client D are marked as dropped out.", {
    event: { name: "CLIENT_DROPOUT", detail: "deadline_no_update" },
  });
  s.coordinator = "failed";
  s.deadline = null;
  push("failed", 2600, "One client's update alone is not a federated model. The round fails instead of publishing it, and v3 stays the global model.", {
    event: { name: "ROUND_FAILED", detail: "quorum_not_met" },
  });
  s.round += 1;
  open(r, ALL, 2000, "The run goes on: the next round starts again from the unchanged model v3.");
  return {
    id: "quorum",
    title: "Too few clients answer",
    problem: "So few updates arrive that the average would mean little.",
    outcome: "The round fails on purpose. The global model stays unchanged and the next round starts from it.",
    frames: r.frames,
  };
}

function lateScenario(): Scenario {
  const r = recorder(ALL, { model: 3, quorum: QUORUM });
  const { s, push } = r;
  opening(r);
  report(r, [A, B, D], 1100, "Client A, Client B and Client D send their updates. Client C is slow and still training.", 0.5);
  s.deadline = 0;
  push("wait", 1800, "The deadline runs out while Client C is still training.");
  s.clients[C] = "dropout";
  push("dropout", 1700, "Client C is marked as dropped out and the round closes.", {
    client: C,
    event: { name: "CLIENT_DROPOUT", detail: "deadline_no_update" },
  });
  aggregate(r, 1500, "FedAvg runs on the three updates that arrived in time.");
  push("done", 1300, "Model v4 is ready.", { event: { name: "ROUND_DONE", detail: "v4" } });
  s.clients[C] = "late";
  push("refused", 2400, "Client C finally sends its update, but it was trained on v3 for a round that is already closed. It is refused.", {
    client: C,
    packets: [{ from: C, to: COORDINATOR, kind: "update", fate: "refused" }],
    event: { name: "UPDATE_REJECTED", detail: "stale_round" },
  });
  s.round += 1;
  open(r, ALL, 2000, "Client C does not resend it. Like the others, it continues from model v4 in the next round.");
  return {
    id: "late",
    title: "An update arrives too late",
    problem: "A slow client finishes after its round has closed.",
    outcome: "The outdated update is refused, so it cannot mix an old model into a newer one.",
    frames: r.frames,
  };
}

function retryScenario(): Scenario {
  const r = recorder(ALL, { model: 3, quorum: QUORUM });
  const { s, push } = r;
  opening(r);
  report(r, [A], 1300, "Client A sends its update and the coordinator accepts it.", 0.6, { database: true });
  push("refused", 2200, "The confirmation is lost on the way back. Client A cannot tell whether its update arrived.", {
    client: A,
    packets: [{ from: COORDINATOR, to: A, kind: "ack", fate: "lost" }],
  });
  s.clients[A] = "sending";
  push("report", 2200, "Client A sends the same update again, with the same update ID.", {
    client: A,
    packets: [{ from: A, to: COORDINATOR, kind: "update" }],
  });
  s.clients[A] = "sent";
  push("refused", 2400, "The coordinator recognises the ID and confirms again. It stores nothing new, so the update still counts once.", {
    client: A,
    packets: [{ from: COORDINATOR, to: A, kind: "ack" }],
    event: { name: "UPDATE_RETRY_ACKNOWLEDGED", detail: "duplicate_idempotent" },
  });
  report(r, [B, C, D], 1100, "The other clients report. The counter says 4 of 4, not 5.", 0.3);
  aggregate(r, 1500, "Client A has exactly the same weight in the average as everyone else.");
  push("done", 1500, "Model v4 is ready.", { event: { name: "ROUND_DONE", detail: "v4" } });
  return {
    id: "retry",
    title: "A confirmation gets lost",
    problem: "The network drops a reply, so a client sends the same update twice.",
    outcome: "Retrying is safe: the update ID makes sure it counts only once.",
    frames: r.frames,
  };
}

function brokenScenario(): Scenario {
  const r = recorder(ALL, { model: 3, quorum: QUORUM });
  const { s, push } = r;
  opening(r);
  report(r, [A, B, C], 1100, "Client A, Client B and Client C send their updates.", 0.5);
  s.clients[D] = "sending";
  push("report", 1800, "Client D sends an update too. Its shape is right, but its numbers have blown up.", {
    client: D,
    packets: [{ from: D, to: COORDINATOR, kind: "broken" }],
  });
  s.clients[D] = "sent";
  s.received = 4;
  s.coordinator = "checking";
  s.deadline = null;
  push("wait", 2000, "Before averaging, the coordinator runs each update through the reference model on test inputs.");
  s.clients[D] = "rejected";
  s.received = 3;
  push("refused", 2400, "Client D's update produces unusable outputs, so it is excluded. One bad update must not ruin the average.", {
    client: D,
    event: { name: "UPDATE_EXCLUDED", detail: "sanity_check_failed" },
  });
  aggregate(r, 1600, "FedAvg runs on the three healthy updates.");
  push("done", 1500, "Model v4 is ready and is not affected by the broken update.", {
    event: { name: "ROUND_DONE", detail: "v4" },
  });
  return {
    id: "broken",
    title: "An update is broken",
    problem: "A client sends weights that look valid but are numerically useless.",
    outcome: "The sanity check excludes that update and the round finishes with the healthy ones.",
    frames: r.frames,
  };
}

function conformanceScenario(): Scenario {
  const r = recorder(ALL, { model: 0, quorum: QUORUM });
  const { s, push } = r;
  s.coordinator = "checking";
  for (const c of ALL) s.clients[c] = "checking";
  push("conformance", 2200, "Before a run, every client reports how it prepares its data. The system compares each report with the run's manifest.", {
    packets: ALL.map((from) => ({ from, to: COORDINATOR, kind: "report" })),
  });
  for (const c of [A, B, C]) s.clients[c] = "idle";
  push("conformance", 1800, "Client A, Client B and Client C match the manifest and become eligible for rounds.", {
    event: { name: "CONFORMANCE_EVALUATED", detail: "passed" },
  });
  s.clients[D] = "blocked";
  push("blocked", 2800, "Client D's labels mean something else: its 0 is the others' 1. Training on that would quietly damage the shared model.", {
    client: D,
    event: { name: "CONFORMANCE_EVALUATED", detail: "label_semantics_mismatch" },
  });
  open(r, [A, B, C], 2000, "The round opens without Client D. It was never selected, so its data cannot affect the model.");
  train(r, [A, B, C], 1400, "Client D can fix its setup and submit a new report. Until then it stays out.", 0.7);
  return {
    id: "conformance",
    title: "A client's data doesn't match",
    problem: "A client encodes its data differently from the others.",
    outcome: "The conformance check keeps it out before it can affect the model.",
    frames: r.frames,
  };
}

export function buildScenarios(): Scenario[] {
  return [
    dropoutScenario(),
    crashScenario(),
    quorumScenario(),
    lateScenario(),
    retryScenario(),
    brokenScenario(),
    conformanceScenario(),
  ];
}
