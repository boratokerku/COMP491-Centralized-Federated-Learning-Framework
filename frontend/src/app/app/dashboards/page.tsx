import type { Metadata } from "next";
import { RunPicker } from "@/components/run-picker";
import { Empty, PageHeader } from "@/components/ui";
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

  // Time charts follow the run: Grafana's default range would squeeze it into a single point.
  const from = run.created_at.getTime() - 5_000;
  const to = run.completed_at ? run.completed_at.getTime() + 15_000 : "now";
  const path = `/grafana/d/cfl-run/cfl-run-overview?orgId=1&var-run_id=${run.run_id}&from=${from}&to=${to}`;
  return (
    <div className="space-y-4">
      <PageHeader title="Dashboards" description="Grafana panels for one run: round state, durations, failures, global and per-client metrics, and communication." />
      <RunPicker base="/app/dashboards" param="run" current={run.run_id} />
      <p className="text-sm">
        <a href={path} className="text-accent underline">
          Open in Grafana
        </a>
      </p>
      <iframe
        key={run.run_id}
        src={`${path}&kiosk&theme=dark`}
        title="Grafana: CFL Run Overview"
        className="h-[2850px] w-full rounded-xl border border-border"
      />
    </div>
  );
}
