import type { Metadata } from "next";
import Link from "next/link";
import { RunPicker } from "@/components/run-picker";
import { Badge, Cell, Empty, Row, Section, Table } from "@/components/ui";
import { requireUser } from "@/lib/auth/guard";
import { fmtBytes, fmtNumber, fmtTime } from "@/lib/format";
import { defaultRunId, getRun, listRounds, listUpdates, parseId } from "@/lib/queries";

export const metadata: Metadata = { title: "Update inbox" };

export default async function Updates({ searchParams }: PageProps<"/app/updates">) {
  await requireUser("/app/updates");
  const q = await searchParams;
  const runId = typeof q.run === "string" ? q.run : await defaultRunId();
  const run = runId ? await getRun(runId) : null;
  if (!run) return <Empty>No runs yet.</Empty>;

  const rounds = await listRounds(run.run_id);
  const wanted = typeof q.round === "string" ? parseId(q.round) : null;
  const round = rounds.find((r) => r.round_id === wanted) ?? rounds.at(-1);
  const inbox = round ? await listUpdates(run.run_id, round.round_id) : null;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Update inbox</h1>
      <RunPicker base="/app/updates" param="run" current={run.run_id} />
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-muted">Round:</span>
        {rounds.map((r) => (
          <Link
            key={r.round_id}
            href={`/app/updates?run=${run.run_id}&round=${r.round_id}`}
            className={`rounded border px-2 py-1 ${
              r.round_id === round?.round_id ? "border-accent" : "border-border"
            }`}
          >
            {r.round_id}
          </Link>
        ))}
      </div>

      {!round || !inbox ? (
        <Empty>This run has no rounds yet.</Empty>
      ) : (
        <>
          <Section title={`Accepted updates, round ${round.round_id}`}>
            {inbox.accepted.length === 0 ? (
              <Empty>No update accepted yet.</Empty>
            ) : (
              <Table head={["Client", "Received", "Model", "Samples", "Size", "SHA-256", "Val F1 before / after"]}>
                {inbox.accepted.map((u) => (
                  <Row key={u.update_id}>
                    <Cell>{u.client_id}</Cell>
                    <Cell>{fmtTime(u.received_at)}</Cell>
                    <Cell>v{u.model_version}</Cell>
                    <Cell>{fmtNumber(u.num_samples)}</Cell>
                    <Cell>{fmtBytes(u.payload_bytes)}</Cell>
                    <Cell mono>{u.checksum_sha256}</Cell>
                    <Cell>
                      {fmtNumber(u.metrics.val_f1_before)} / {fmtNumber(u.metrics.val_f1_after)}
                    </Cell>
                  </Row>
                ))}
              </Table>
            )}
          </Section>
          <Section title="Rejected uploads">
            {inbox.rejected.length === 0 ? (
              <Empty>No update was rejected in this round.</Empty>
            ) : (
              <Table head={["Time", "Client", "Event", "Reason", "Bytes"]}>
                {inbox.rejected.map((e) => (
                  <Row key={e.event_id}>
                    <Cell>{fmtTime(e.occurred_at)}</Cell>
                    <Cell>{e.client_id ?? "-"}</Cell>
                    <Cell>
                      <Badge value="REJECTED" /> <span className="font-mono text-xs">{e.event_type}</span>
                    </Cell>
                    <Cell>{e.reason_code ?? "-"}</Cell>
                    <Cell>{fmtBytes(e.message_bytes)}</Cell>
                  </Row>
                ))}
              </Table>
            )}
          </Section>
        </>
      )}
    </div>
  );
}
