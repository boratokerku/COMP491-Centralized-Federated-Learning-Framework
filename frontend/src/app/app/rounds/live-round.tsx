"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { Badge, Cell, Empty, Row, Stat, Stats, Table } from "@/components/ui";
import { fmtNumber, fmtRemaining, fmtTime } from "@/lib/format";

type RoundSummary = {
  round_id: number;
  state: string;
  min_participants: number;
  selected_count: number;
  accepted_count: number;
  opened_at: string;
  deadline_at: string;
  closed_at: string | null;
  failure_reason: string | null;
};
type Participant = {
  client_id: string;
  status: string;
  reason_code: string | null;
  finished_at: string | null;
};
type Detail = RoundSummary & {
  participants: Participant[];
  metrics: { metric_name: string; value: number; unit: string }[];
};

const OPEN_STATES = ["OPEN", "COLLECTING", "AGGREGATING"];
const POLL_MS = 3000;

export function LiveRound({
  runId,
  runStatus,
  initial,
}: {
  runId: string;
  runStatus: string;
  initial: Detail | null;
}) {
  const router = useRouter();
  const [detail, setDetail] = useState<Detail | null>(initial);
  const [now, setNow] = useState(() => Date.now());
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      const list = await fetch(`/api/v1/runs/${runId}/rounds`, { cache: "no-store" });
      if (list.status === 401) return router.replace("/login?next=/app/rounds");
      if (!list.ok) throw new Error(`rounds: HTTP ${list.status}`);
      const rounds: RoundSummary[] = (await list.json()).rounds;
      const latest = rounds.at(-1);
      if (!latest) return setDetail(null);
      const res = await fetch(`/api/v1/runs/${runId}/rounds/${latest.round_id}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`round ${latest.round_id}: HTTP ${res.status}`);
      setDetail(await res.json());
      setError(null);
      setUpdatedAt(Date.now());
    } catch (e) {
      setError(e instanceof Error ? e.message : "update failed");
    }
  }, [runId, router]);

  useEffect(() => {
    const tick = setInterval(() => setNow(Date.now()), 1000);
    const poll = setInterval(() => {
      if (document.visibilityState === "visible") void refresh();
    }, POLL_MS);
    return () => {
      clearInterval(tick);
      clearInterval(poll);
    };
  }, [refresh]);

  if (!detail) return <Empty>This run has no rounds yet.</Empty>;

  const live = OPEN_STATES.includes(detail.state);
  const quorum = detail.accepted_count >= detail.min_participants;
  return (
    <div className="space-y-5" aria-live="polite">
      <Stats>
        <Stat label="Round">{detail.round_id}</Stat>
        <Stat label="State">
          <Badge value={detail.state} />
        </Stat>
        <Stat
          label={live ? "Time left" : "Closed"}
          hint={live ? `deadline ${fmtTime(detail.deadline_at)}` : undefined}
        >
          {live ? fmtRemaining(new Date(detail.deadline_at).getTime() - now) : fmtTime(detail.closed_at)}
        </Stat>
        <Stat
          label="Updates received"
          hint={`${detail.min_participants} needed${live ? (quorum ? ", quorum reached" : ", quorum not reached yet") : ""}`}
        >
          {detail.accepted_count} of {detail.selected_count}
        </Stat>
      </Stats>

      <p className="text-xs text-muted">
        {runStatus === "RUNNING" ? "Live: refreshes every 3 seconds" : `The run is ${runStatus.toLowerCase()}`}
        {updatedAt ? `, last refresh ${fmtTime(new Date(updatedAt))}` : ""}
        {detail.failure_reason ? `. The round failed: ${detail.failure_reason}` : ""}
      </p>

      {error && (
        <p role="alert" className="text-sm text-red-600">
          Could not refresh ({error}). Showing the last data received.
        </p>
      )}

      <Table head={["Client", "Status", "Reason", "Finished"]}>
        {detail.participants.map((p) => (
          <Row key={p.client_id}>
            <Cell>{p.client_id}</Cell>
            <Cell>
              <Badge value={p.status} />
            </Cell>
            <Cell>{p.reason_code ?? "-"}</Cell>
            <Cell>{fmtTime(p.finished_at)}</Cell>
          </Row>
        ))}
      </Table>

      {detail.metrics.length > 0 && (
        <details className="text-sm">
          <summary className="cursor-pointer text-muted">Round metrics</summary>
          <Table head={["Metric", "Value", "Unit"]}>
            {detail.metrics.map((m) => (
              <Row key={m.metric_name}>
                <Cell mono>{m.metric_name}</Cell>
                <Cell>{fmtNumber(m.value)}</Cell>
                <Cell>{m.unit}</Cell>
              </Row>
            ))}
          </Table>
        </details>
      )}
    </div>
  );
}
