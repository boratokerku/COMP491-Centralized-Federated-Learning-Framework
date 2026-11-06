import type { Metadata } from "next";
import { Empty, PageHeader } from "@/components/ui";
import { RunPicker } from "@/components/run-picker";
import { requireUser } from "@/lib/auth/guard";
import { defaultRunId, getRound, getRun, listRounds } from "@/lib/queries";
import { LiveRound } from "./live-round";

export const metadata: Metadata = { title: "Live round" };

export default async function Rounds({ searchParams }: PageProps<"/app/rounds">) {
  await requireUser("/app/rounds");
  const wanted = (await searchParams).run;
  const runId = typeof wanted === "string" ? wanted : await defaultRunId();
  const run = runId ? await getRun(runId) : null;
  if (!run) return <Empty>No runs yet. Start one and it shows up here.</Empty>;

  const latest = (await listRounds(run.run_id)).at(-1);
  const detail = latest ? await getRound(run.run_id, latest.round_id) : null;
  return (
    <div className="space-y-5">
      <PageHeader title="Live round" description="Follow the current round of a run as it happens: who has reported, who dropped out, and when the round closes." />
      <RunPicker base="/app/rounds" param="run" current={run.run_id} />
      <LiveRound
        key={run.run_id}
        runId={run.run_id}
        runStatus={run.status}
        initial={detail ? JSON.parse(JSON.stringify(detail)) : null}
      />
    </div>
  );
}
