import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Badge, Cell, Empty, Facts, Row, Section, Table } from "@/components/ui";
import { requireUser } from "@/lib/auth/guard";
import { fmtNumber, fmtTime, shortId } from "@/lib/format";
import { getClient } from "@/lib/queries";

export const metadata: Metadata = { title: "Client" };

export default async function ClientDetail({ params }: PageProps<"/app/clients/[clientId]">) {
  const clientId = decodeURIComponent((await params).clientId);
  await requireUser(`/app/clients/${encodeURIComponent(clientId)}`);
  const client = await getClient(clientId);
  if (!client) notFound();

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold">{client.client_id}</h1>
      <Facts
        items={[
          ["Status", <Badge key="s" value={client.status} />],
          ["Client version", client.client_version ?? "-"],
          ["Protocol", client.protocol_version ?? "-"],
          ["Approved", fmtTime(client.created_at)],
          ["Registered", fmtTime(client.registered_at)],
          ["Last seen", fmtTime(client.last_seen_at)],
        ]}
      />

      <Section title="Conformance attempts">
        {client.conformance.length === 0 ? (
          <Empty>No conformance report yet.</Empty>
        ) : (
          <Table head={["Checked", "Manifest", "Client version", "Result", "Reason", "Details"]}>
            {client.conformance.map((c) => (
              <Row key={c.report_id}>
                <Cell>{fmtTime(c.checked_at)}</Cell>
                <Cell>{c.manifest_version}</Cell>
                <Cell>{c.client_version}</Cell>
                <Cell>
                  <Badge value={c.status} />
                </Cell>
                <Cell>{c.reason_code ?? "-"}</Cell>
                <Cell>
                  <details>
                    <summary className="cursor-pointer text-muted">report</summary>
                    <pre className="mt-1 max-w-md overflow-x-auto text-xs">{JSON.stringify(c.report_payload, null, 2)}</pre>
                  </details>
                </Cell>
              </Row>
            ))}
          </Table>
        )}
      </Section>

      <Section title="Participation">
        {client.participation.length === 0 ? (
          <Empty>This client has not been selected for a round.</Empty>
        ) : (
          <Table head={["Run", "Round", "Status", "Reason", "Finished"]}>
            {client.participation.map((p) => (
              <Row key={`${p.run_id}-${p.round_id}`}>
                <Cell mono>
                  <Link href={`/app/runs/${p.run_id}`} className="text-accent underline">
                    {shortId(p.run_id)}
                  </Link>
                </Cell>
                <Cell>{p.round_id}</Cell>
                <Cell>
                  <Badge value={p.status} />
                </Cell>
                <Cell>{p.reason_code ?? "-"}</Cell>
                <Cell>{fmtTime(p.finished_at)}</Cell>
              </Row>
            ))}
          </Table>
        )}
      </Section>

      <Section title="Reported metrics (latest 500)">
        {client.metrics.length === 0 ? (
          <Empty>No metrics reported.</Empty>
        ) : (
          <Table head={["Run", "Round", "Metric", "Value", "Unit"]}>
            {client.metrics.map((m, i) => (
              <Row key={i}>
                <Cell mono>{shortId(m.run_id)}</Cell>
                <Cell>{m.round_id}</Cell>
                <Cell mono>{m.metric_name}</Cell>
                <Cell>{fmtNumber(m.value)}</Cell>
                <Cell>{m.unit}</Cell>
              </Row>
            ))}
          </Table>
        )}
      </Section>
    </div>
  );
}
