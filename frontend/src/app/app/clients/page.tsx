import type { Metadata } from "next";
import Link from "next/link";
import { Badge, Cell, Empty, Row, Table } from "@/components/ui";
import { requireUser } from "@/lib/auth/guard";
import { fmtTime } from "@/lib/format";
import { listClients } from "@/lib/queries";

export const metadata: Metadata = { title: "Clients" };

export default async function Clients() {
  await requireUser("/app/clients");
  const clients = await listClients();
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Clients and conformance</h1>
      {clients.length === 0 ? (
        <Empty>No client approved yet (cfl-admin add-client).</Empty>
      ) : (
        <Table head={["Client", "Status", "Version", "Last seen", "Conformance", "Manifest", "Reason", "Checked"]}>
          {clients.map((c) => (
            <Row key={c.client_id}>
              <Cell>
                <Link href={`/app/clients/${encodeURIComponent(c.client_id)}`} className="text-accent underline">
                  {c.client_id}
                </Link>
              </Cell>
              <Cell>
                <Badge value={c.status} />
              </Cell>
              <Cell>{c.client_version ?? "-"}</Cell>
              <Cell>{fmtTime(c.last_seen_at)}</Cell>
              <Cell>
                <Badge value={c.latest_conformance?.status} />
              </Cell>
              <Cell>{c.latest_conformance?.manifest_version ?? "-"}</Cell>
              <Cell>{c.latest_conformance?.reason_code ?? "-"}</Cell>
              <Cell>{fmtTime(c.latest_conformance?.checked_at)}</Cell>
            </Row>
          ))}
        </Table>
      )}
    </div>
  );
}
