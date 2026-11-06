import type { Metadata } from "next";
import Link from "next/link";
import { Badge, Cell, Empty, PageHeader, Row, Table } from "@/components/ui";
import { requireUser } from "@/lib/auth/guard";
import { fmtTime, shortId } from "@/lib/format";
import { listRuns } from "@/lib/queries";
import { hasReport } from "@/lib/reports";

export const metadata: Metadata = { title: "Reports" };

export default async function Reports() {
  await requireUser("/app/reports");
  const { runs } = await listRuns({ limit: 100, offset: 0 });
  const withReport = await Promise.all(runs.map(async (r) => ({ ...r, report: await hasReport(r.run_id) })));
  return (
    <div className="space-y-4">
      <PageHeader title="Reports" description="The report and plots that are written when a run is finished. Open one to read it or download the files." />
      {runs.length === 0 ? (
        <Empty>No runs yet.</Empty>
      ) : (
        <Table head={["Run", "Name", "Status", "Completed", "Report"]}>
          {withReport.map((r) => (
            <Row key={r.run_id}>
              <Cell mono>{shortId(r.run_id)}</Cell>
              <Cell>{r.name}</Cell>
              <Cell>
                <Badge value={r.status} />
              </Cell>
              <Cell>{fmtTime(r.completed_at)}</Cell>
              <Cell>
                {r.report ? (
                  <Link href={`/app/reports/${r.run_id}`} className="text-accent underline">
                    Open
                  </Link>
                ) : (
                  <span className="text-muted">not generated (cfl-report {shortId(r.run_id)}…)</span>
                )}
              </Cell>
            </Row>
          ))}
        </Table>
      )}
    </div>
  );
}
