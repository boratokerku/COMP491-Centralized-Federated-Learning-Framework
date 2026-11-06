import type { Metadata } from "next";
import { Card, PageHeader } from "@/components/ui";
import { DemoPlayer } from "@/components/demo-player";
import demoRun from "@/content/demo-run.json";
import type { DemoRun } from "@/lib/demo-frames";

export const metadata: Metadata = {
  title: "How it works",
  description: "A replay of a federated training run, and how the framework reacts to each kind of failure.",
};

const demo = demoRun as DemoRun;

export default function Demo() {
  const scripted = demo.events.filter((e) => e.scripted);
  return (
    <div className="space-y-8">
      <PageHeader
        title="How it works"
        description={`A replay of one finished training run: ${demo.clients.length} clients, ${demo.rounds.length} rounds, a text classification task. In each round the clients train on their own data and send back only model updates; the coordinator combines them into a new global model. Below it, each failure can be watched on its own.`}
      />

      <DemoPlayer demo={demo} />

      <Card>
        <h2 className="font-semibold">About this replay</h2>
        <ul className="mt-2 list-disc space-y-1.5 pl-5 text-sm text-muted">
          <li>
            The curve and the per-round numbers come from the report of a real run of the
            framework (a local simulation on the SMS Spam data).
          </li>
          {scripted.length > 0 && (
            <li>
              A local simulation cannot lose a client or crash, so the{" "}
              {scripted
                .map((e) =>
                  e.type === "CLIENT_DROPOUT"
                    ? `dropout in round ${e.round}`
                    : `coordinator restart in round ${e.round}`,
                )
                .join(" and the ")}{" "}
              are scripted to show how the real system reacts. Client names are made up.
            </li>
          )}
          <li>
            The failure scenarios are scripted from the rules the framework applies. Their
            steps, event names and reason codes match the system, but they are not recordings
            of a run and their model versions are only examples.
          </li>
          <li>
            This page does not connect to a running system, and visitors cannot start runs.
          </li>
        </ul>
      </Card>
    </div>
  );
}
