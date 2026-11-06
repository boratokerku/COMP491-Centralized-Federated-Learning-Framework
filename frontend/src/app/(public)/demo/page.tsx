import type { Metadata } from "next";
import { Card, PageHeader } from "@/components/ui";
import { DemoPlayer } from "@/components/demo-player";
import demoRun from "@/content/demo-run.json";
import type { DemoRun } from "@/lib/demo-frames";

export const metadata: Metadata = {
  title: "How it works",
  description: "A replay of a federated training run: clients report, one drops out, the coordinator restarts.",
};

const demo = demoRun as DemoRun;

export default function Demo() {
  const scripted = demo.events.filter((e) => e.scripted);
  return (
    <div className="space-y-8">
      <PageHeader
        title="How it works"
        description={`A replay of one finished training run: ${demo.clients.length} clients, ${demo.rounds.length} rounds, a text classification task. In each round the clients train on their own data and send back only model updates; the coordinator combines them into a new global model. Watch what happens when a client drops out and when the coordinator restarts.`}
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
            This page is a recording. It does not connect to a running system, and visitors
            cannot start runs.
          </li>
        </ul>
      </Card>
    </div>
  );
}
