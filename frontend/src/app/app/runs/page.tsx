import type { Metadata } from "next";
import Link from "next/link";
import { Badge, Cell, Empty, PageHeader, Row, Table } from "@/components/ui";
import { requireUser } from "@/lib/auth/guard";
import { fmtTime, shortId } from "@/lib/format";
import { listRuns } from "@/lib/queries";

export const metadata: Metadata = { title: "Runs" };

export default async function Runs() {
  await requireUser("/app/runs");
  const { total, runs } = await listRuns({ limit: 100, offset: 0 });
  return (
    <div className="space-y-4">
      <PageHeader title="Runs" description="Every training run, newest first. Open one for its configuration, versions, metric charts and final model." />
      {runs.length === 0 ? (
        <Empty>No runs yet.</Empty>
      ) : (
        <>
          <p className="text-sm text-muted">{total} in total</p>
          <Table head={["Run", "Name", "Status", "Strategy", "Task", "Model", "Started", "Completed"]}>
            {runs.map((r) => (
              <Row key={r.run_id}>
                <Cell mono>
                  <Link href={`/app/runs/${r.run_id}`} className="text-accent underline">
                    {shortId(r.run_id)}
                  </Link>
                </Cell>
                <Cell>{r.name}</Cell>
                <Cell>
                  <Badge value={r.status} />
                </Cell>
                <Cell>{r.strategy}</Cell>
                <Cell>{r.task_name}</Cell>
                <Cell>v{r.model_version}</Cell>
                <Cell>{fmtTime(r.started_at)}</Cell>
                <Cell>{fmtTime(r.completed_at)}</Cell>
              </Row>
            ))}
          </Table>
        </>
      )}
    </div>
  );
}
