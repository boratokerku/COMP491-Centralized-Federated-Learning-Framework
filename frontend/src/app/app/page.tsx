import type { Metadata } from "next";
import Link from "next/link";
import {
  Badge,
  Card,
  Cell,
  Empty,
  PageHeader,
  Row,
  Section,
  Stat,
  Stats,
  Table,
} from "@/components/ui";
import { requireUser } from "@/lib/auth/guard";
import { fmtTime } from "@/lib/format";
import { defaultRunId, getRun, listClients, listEvents, listRounds, listRuns } from "@/lib/queries";

export const metadata: Metadata = { title: "Overview" };

const areas = [
  ["/app/rounds", "Live round", "Follow the current round as it happens."],
  ["/app/runs", "Runs", "Configuration, versions, charts and the final model of every run."],
  ["/app/dashboards", "Dashboards", "Grafana panels for durations, failures and transfers."],
  ["/app/updates", "Update inbox", "Accepted and rejected client updates of a round."],
  ["/app/logs", "Event log", "Everything that happened, filterable."],
  ["/app/clients", "Clients", "Who is registered and whether they are compatible."],
  ["/app/reports", "Reports", "The report and plots written at the end of a run."],
] as const;

export default async function Overview() {
  const user = await requireUser("/app");
  const [{ total, runs }, clients, { events }] = await Promise.all([
    listRuns({ limit: 100, offset: 0 }),
    listClients(),
    listEvents({ limit: 6, offset: 0 }),
  ]);
  const runId = await defaultRunId();
  const run = runId ? await getRun(runId) : null;
  const rounds = run ? await listRounds(run.run_id) : [];
  const current = rounds.at(-1);
  const running = runs.filter((r) => r.status === "RUNNING").length;
  const passed = clients.filter((c) => c.latest_conformance?.status === "PASSED").length;
  const failedConformance = clients.filter((c) => c.latest_conformance?.status === "FAILED");
  const failedRounds = rounds.filter((r) => r.state === "FAILED").length;

  return (
    <>
      <PageHeader title={`Welcome, ${user}`} description="What is happening in the system right now, and where to look for more." />

      <Stats>
        <Stat label="Runs" hint={`${running} running`}>
          {total}
        </Stat>
        <Stat label="Clients compatible" hint={`${clients.length - passed} not (yet)`}>
          {passed} of {clients.length}
        </Stat>
        <Stat label="Latest run">{run ? <Badge value={run.status} /> : "-"}</Stat>
        <Stat label="Current round" hint={current ? `${current.accepted_count} of ${current.selected_count} updates` : undefined}>
          {current ? `${current.round_id} ` : "-"}
          {current && <Badge value={current.state} />}
        </Stat>
      </Stats>

      {run ? (
        <Card>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="font-semibold">{run.name}</h2>
              <p className="text-sm text-muted">
                {run.strategy} on {run.task_name}, model v{run.model_version}, started {fmtTime(run.started_at)}
              </p>
            </div>
            <div className="flex gap-2 text-sm">
              <Link href={`/app/rounds?run=${run.run_id}`} className="rounded-lg bg-accent px-3 py-1.5 font-medium text-background hover:opacity-90">
                Live round
              </Link>
              <Link href={`/app/runs/${run.run_id}`} className="rounded-lg border border-border px-3 py-1.5 hover:bg-surface">
                Run details
              </Link>
            </div>
          </div>
        </Card>
      ) : (
        <Empty>No run yet. When the coordinator starts one, it shows up here.</Empty>
      )}

      {(failedRounds > 0 || failedConformance.length > 0) && (
        <Section title="Needs attention">
          <ul className="space-y-2 text-sm">
            {failedRounds > 0 && (
              <li className="rounded-lg border border-danger bg-danger-soft p-3">
                {failedRounds} failed round{failedRounds > 1 ? "s" : ""} in the latest run.{" "}
                <Link href={`/app/runs/${run?.run_id}`} className="underline">
                  See why
                </Link>
              </li>
            )}
            {failedConformance.map((c) => (
              <li key={c.client_id} className="rounded-lg border border-danger bg-danger-soft p-3">
                {c.client_id} failed the conformance check ({c.latest_conformance?.reason_code ?? "no reason recorded"}).{" "}
                <Link href={`/app/clients/${encodeURIComponent(c.client_id)}`} className="underline">
                  Details
                </Link>
              </li>
            ))}
          </ul>
        </Section>
      )}

      <Section title="Recent events" hint="The latest six. The event log has the rest.">
        {events.length === 0 ? (
          <Empty>Nothing has happened yet.</Empty>
        ) : (
          <Table head={["Time", "Event", "Round", "Client", "Reason"]}>
            {events.map((e) => (
              <Row key={e.event_id}>
                <Cell>{fmtTime(e.occurred_at)}</Cell>
                <Cell mono>{e.event_type}</Cell>
                <Cell>{e.round_id ?? "-"}</Cell>
                <Cell>{e.client_id ?? "-"}</Cell>
                <Cell>{e.reason_code ?? "-"}</Cell>
              </Row>
            ))}
          </Table>
        )}
      </Section>

      <Section title="Where to look">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {areas.map(([href, title, text]) => (
            <Link key={href} href={href} className="rounded-xl border border-border p-4 hover:bg-surface">
              <div className="font-medium">{title}</div>
              <div className="mt-1 text-sm text-muted">{text}</div>
            </Link>
          ))}
        </div>
      </Section>
    </>
  );
}
