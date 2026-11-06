import type { Metadata } from "next";
import Link from "next/link";
import { Cell, Empty, PageHeader, Row, Table } from "@/components/ui";
import { requireUser } from "@/lib/auth/guard";
import { fmtTime, shortId } from "@/lib/format";
import { eventTypes, isUuid, listEvents, parseId } from "@/lib/queries";

export const metadata: Metadata = { title: "Event log" };
const PAGE = 50;

export default async function Logs({ searchParams }: PageProps<"/app/logs">) {
  await requireUser("/app/logs");
  const q = await searchParams;
  const get = (key: string) => (typeof q[key] === "string" && q[key] ? (q[key] as string) : undefined);

  const runId = get("run_id");
  const round = get("round_id");
  const offset = Math.max(0, Number(get("offset") ?? 0) || 0);
  const filter = {
    run_id: runId && isUuid(runId) ? runId : undefined,
    round_id: round ? (parseId(round) ?? undefined) : undefined,
    client_id: get("client_id"),
    event_type: get("event_type"),
    status: get("status"),
  };
  const { total, events } = await listEvents({ ...filter, limit: PAGE, offset });
  const types = await eventTypes();

  const params = new URLSearchParams(
    Object.entries({ run_id: runId, round_id: round, client_id: filter.client_id, event_type: filter.event_type, status: filter.status })
      .filter((e): e is [string, string] => Boolean(e[1])),
  );
  const link = (o: number) => `/app/logs?${new URLSearchParams([...params, ["offset", String(o)]])}`;
  const field = "rounded border border-border bg-transparent px-2 py-1";

  return (
    <div className="space-y-5">
      <PageHeader title="Event log" description="Everything that happened in the system. Narrow it down with the filters." />
      <form className="flex flex-wrap items-end gap-3 text-sm">
        <label className="space-y-1">
          <span className="block text-muted">Run ID</span>
          <input name="run_id" defaultValue={runId} className={`${field} w-72`} />
        </label>
        <label className="space-y-1">
          <span className="block text-muted">Round</span>
          <input name="round_id" defaultValue={round} className={`${field} w-20`} />
        </label>
        <label className="space-y-1">
          <span className="block text-muted">Client</span>
          <input name="client_id" defaultValue={filter.client_id} className={`${field} w-32`} />
        </label>
        <label className="space-y-1">
          <span className="block text-muted">Event type</span>
          <select name="event_type" defaultValue={filter.event_type ?? ""} className={field}>
            <option value="">all</option>
            {types.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
        </label>
        <label className="space-y-1">
          <span className="block text-muted">Status</span>
          <input name="status" defaultValue={filter.status} className={`${field} w-28`} />
        </label>
        <button type="submit" className="rounded bg-accent px-3 py-1 text-white">
          Filter
        </button>
        <Link href="/app/logs" className="py-1 text-muted underline">
          Reset
        </Link>
      </form>

      {runId && !filter.run_id && <p className="text-sm text-red-600">Run ID is not a valid UUID, ignored.</p>}
      {events.length === 0 ? (
        <Empty>No events match.</Empty>
      ) : (
        <>
          <p className="text-sm text-muted">
            {offset + 1}-{offset + events.length} of {total}
          </p>
          <Table head={["Time", "Event", "Run", "Round", "Client", "Reason", "Details"]}>
            {events.map((e) => (
              <Row key={e.event_id}>
                <Cell>{fmtTime(e.occurred_at)}</Cell>
                <Cell mono>{e.event_type}</Cell>
                <Cell mono>{e.run_id ? shortId(e.run_id) : "-"}</Cell>
                <Cell>{e.round_id ?? "-"}</Cell>
                <Cell>{e.client_id ?? "-"}</Cell>
                <Cell>{e.reason_code ?? "-"}</Cell>
                <Cell>
                  {Object.keys(e.details).length ? (
                    <details>
                      <summary className="cursor-pointer text-muted">show</summary>
                      <pre className="mt-1 max-w-md overflow-x-auto text-xs">{JSON.stringify(e.details, null, 2)}</pre>
                    </details>
                  ) : (
                    "-"
                  )}
                </Cell>
              </Row>
            ))}
          </Table>
          <div className="flex gap-4 text-sm">
            {offset > 0 && (
              <Link href={link(Math.max(0, offset - PAGE))} className="text-accent underline">
                Newer
              </Link>
            )}
            {offset + PAGE < total && (
              <Link href={link(offset + PAGE)} className="text-accent underline">
                Older
              </Link>
            )}
          </div>
        </>
      )}
    </div>
  );
}
