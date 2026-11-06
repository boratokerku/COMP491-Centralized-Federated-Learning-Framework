// The replay on the public "how it works" page, as a list of frames. Kept free of React and
// of imports so it can be tested with `npm test`.

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

export type ClientState = "idle" | "training" | "sent" | "dropout";
export type CoordinatorState = "waiting" | "collecting" | "down" | "aggregating" | "done";

export type Frame = {
  kind: "open" | "report" | "dropout" | "crash" | "resume" | "aggregate" | "done";
  round: number;
  /** The client this frame is about (report and dropout frames). */
  client?: string;
  clients: Record<string, ClientState>;
  coordinator: CoordinatorState;
  /** Rounds whose global model is already on the curve. */
  completed: number;
  caption: string;
};

export function buildFrames(demo: DemoRun): Frame[] {
  const frames: Frame[] = [];
  let completed = 0;
  const idle = () => Object.fromEntries(demo.clients.map((c) => [c, "idle" as ClientState]));

  for (const r of demo.rounds) {
    const clients = idle();
    const push = (kind: Frame["kind"], coordinator: CoordinatorState, caption: string, client?: string) =>
      frames.push({ kind, round: r.round, client, clients: { ...clients }, coordinator, completed, caption });

    for (const c of demo.clients) clients[c] = "training";
    push("open", "collecting", `Round ${r.round}: the coordinator sends the current global model to the clients. Each trains on its own data.`);

    for (const p of r.participants) {
      if (p.status === "UPDATE_ACCEPTED") {
        clients[p.client] = "sent";
        push("report", "collecting", `${p.client} sends only its model update. Its data stays on its own machine.`, p.client);
      } else {
        clients[p.client] = "dropout";
        push("dropout", "collecting", `${p.client} does not answer before the deadline. The round continues without it.`, p.client);
      }
    }

    if (demo.events.some((e) => e.round === r.round && e.type === "COORDINATOR_RESTART")) {
      push("crash", "down", "The coordinator process stops. The state of the round is kept in the database.");
      push("resume", "collecting", "The coordinator restarts, reads the round from the database and carries on. No update is lost.");
    }

    push("aggregate", "aggregating", "The coordinator combines the received updates into a new global model (FedAvg).");
    completed = r.round;
    push("done", "done", `The new global model is evaluated: F1 ${r.f1.toFixed(2)}, ROC-AUC ${r.rocAuc.toFixed(2)}.`);
  }
  return frames;
}
