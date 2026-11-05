import type { Metadata } from "next";
import { RunPicker } from "@/components/run-picker";
import { Empty } from "@/components/ui";
import { requireUser } from "@/lib/auth/guard";
import { defaultRunId, getRun } from "@/lib/queries";

export const metadata: Metadata = { title: "Dashboards" };

// Dashboard "CFL Run Overview" (deploy/grafana/provisioning/dashboards/cfl-run.json). Grafana
// is served by Caddy under /grafana/ and only answers requests that carry a website session.
export default async function Dashboards({ searchParams }: PageProps<"/app/dashboards">) {
  await requireUser("/app/dashboards");
  const wanted = (await searchParams).run;
  const runId = typeof wanted === "string" ? wanted : await defaultRunId();
  const run = runId ? await getRun(runId) : null;
  if (!run) return <Empty>No runs yet.</Empty>;

  const path = `/grafana/d/cfl-run/cfl-run-overview?orgId=1&var-run_id=${run.run_id}`;
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Dashboards</h1>
      <RunPicker base="/app/dashboards" param="run" current={run.run_id} />
      <p className="text-sm">
        <a href={path} className="text-accent underline">
          Open in Grafana
        </a>
      </p>
      <iframe
        key={run.run_id}
        src={`${path}&kiosk`}
        title="Grafana: CFL Run Overview"
        className="h-[1500px] w-full rounded border border-border"
      />
    </div>
  );
}
