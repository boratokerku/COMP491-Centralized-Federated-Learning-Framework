import { guard, notFound } from "@/lib/api";
import { getRun, listRounds } from "@/lib/queries";

export async function GET(_: Request, ctx: RouteContext<"/api/v1/runs/[runId]/rounds">) {
  const denied = await guard();
  if (denied) return denied;
  const { runId } = await ctx.params;
  if (!(await getRun(runId))) return notFound("run");
  return Response.json({ run_id: runId, rounds: await listRounds(runId) });
}
