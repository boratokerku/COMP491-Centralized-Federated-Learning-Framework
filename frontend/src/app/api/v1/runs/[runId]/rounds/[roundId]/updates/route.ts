import { guard, notFound } from "@/lib/api";
import { getRound, listUpdates, parseId } from "@/lib/queries";

export async function GET(
  _: Request,
  ctx: RouteContext<"/api/v1/runs/[runId]/rounds/[roundId]/updates">,
) {
  const denied = await guard();
  if (denied) return denied;
  const { runId, roundId } = await ctx.params;
  const id = parseId(roundId);
  if (id === null || !(await getRound(runId, id))) return notFound("round");
  return Response.json(await listUpdates(runId, id));
}
