import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { LineChart, type Series } from "@/components/line-chart";
import { Badge, Cell, Facts, Row, Section, Table } from "@/components/ui";
import { requireUser } from "@/lib/auth/guard";
import { fmtBytes, fmtTime } from "@/lib/format";
import { getRun, listRounds, roundMetricSeries } from "@/lib/queries";
import { hasReport } from "@/lib/reports";

export const metadata: Metadata = { title: "Run" };

const GLOBAL = [
  ["global_f1", "F1"],
  ["global_roc_auc", "ROC-AUC"],
  ["global_pr_auc", "PR-AUC"],
] as const;
const TIMING = [
  ["round_duration_ms", "round duration"],
  ["waiting_ms", "waiting for clients"],
  ["aggregation_ms", "aggregation"],
  ["evaluation_ms", "evaluation"],
] as const;

export default async function RunDetail({ params }: PageProps<"/app/runs/[runId]">) {
  const { runId } = await params;
  await requireUser(`/app/runs/${runId}`);
  const run = await getRun(runId);
  if (!run) notFound();

  const rounds = await listRounds(runId);
  const names = [...GLOBAL, ...TIMING, ["model_bytes", "model bytes"] as const].map((m) => m[0]);
  const rows = await roundMetricSeries(runId, names);
  const series = (defs: readonly (readonly [string, string])[]): Series[] =>
    defs.map(([name, label]) => ({
      label,
      points: rows.filter((r) => r.metric_name === name).map((r) => ({ x: r.round_id, y: r.value })),
    }));
  const report = await hasReport(runId);

  return (
    <div className="space-y-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold">{run.name}</h1>
        <p className="font-mono text-xs text-muted">{run.run_id}</p>
      </div>

      <Section title="Overview">
        <Facts
          items={[
            ["Status", <Badge key="s" value={run.status} />],
            ["Strategy", run.strategy],
            ["Task", run.task_name],
            ["Seed", String(run.seed)],
            ["Created", fmtTime(run.created_at)],
            ["Started", fmtTime(run.started_at)],
            ["Completed", fmtTime(run.completed_at)],
            ...(run.failure_reason ? ([["Failure reason", run.failure_reason]] as [string, string][]) : []),
          ]}
        />
      </Section>

      <Section title="Versions">
        <Facts
          items={[
            ["Protocol", run.protocol_version],
            ["Manifest", `${run.manifest_version} (${run.manifest_sha256})`],
            ["Preprocessing", run.preprocessing_version],
            ["Model architecture", `${run.model_architecture} ${run.model_architecture_version}`],
            ["Model fingerprint", run.model_fingerprint],
          ]}
        />
      </Section>

      <Section title="Final model">
        {run.final_model_path ? (
          <Facts
            items={[
              ["Version", `v${run.model_version}`],
              ["Path", run.final_model_path],
              ["SHA-256", run.final_model_sha256],
              ["Format", run.final_model_format],
              ["Size", fmtBytes(run.final_model_bytes)],
            ]}
          />
        ) : (
          <p className="text-sm text-muted">Not produced yet (current model v{run.model_version}).</p>
        )}
      </Section>

      <Section title="Rounds">
        <Table head={["Round", "State", "Updates", "Min", "Model", "Opened", "Closed", "Reason"]}>
          {rounds.map((r) => (
            <Row key={r.round_id}>
              <Cell>{r.round_id}</Cell>
              <Cell>
                <Badge value={r.state} />
              </Cell>
              <Cell>
                {r.accepted_count}/{r.selected_count}
              </Cell>
              <Cell>{r.min_participants}</Cell>
              <Cell>{r.output_model_version === null ? "-" : `v${r.output_model_version}`}</Cell>
              <Cell>{fmtTime(r.opened_at)}</Cell>
              <Cell>{fmtTime(r.closed_at)}</Cell>
              <Cell>{r.failure_reason ?? "-"}</Cell>
            </Row>
          ))}
        </Table>
        <p className="text-sm">
          <Link href={`/app/updates?run=${runId}`} className="text-accent underline">
            Update inbox
          </Link>
          {" · "}
          <Link href={`/app/logs?run_id=${runId}`} className="text-accent underline">
            Event log
          </Link>
          {" · "}
          {report ? (
            <Link href={`/app/reports/${runId}`} className="text-accent underline">
              Report
            </Link>
          ) : (
            <span className="text-muted">Report not generated yet (cfl-report {runId})</span>
          )}
        </p>
      </Section>

      <Section title="Metric panels">
        <div className="grid gap-6 lg:grid-cols-2">
          <LineChart title="Global model on the held-out test set" series={series(GLOBAL)} yLabel="score" yMin={0} yMax={1} />
          <LineChart title="Round duration (ms)" series={series(TIMING)} yLabel="ms" />
          <LineChart title="Model size (bytes)" series={series([["model_bytes", "model"]])} yLabel="bytes" />
        </div>
      </Section>

      <Section title="Configuration">
        <p className="font-mono text-xs text-muted">SHA-256 {run.config_sha256}</p>
        <pre className="overflow-x-auto rounded border border-border p-3 text-xs">
          {JSON.stringify(run.config, null, 2)}
        </pre>
      </Section>
    </div>
  );
}
