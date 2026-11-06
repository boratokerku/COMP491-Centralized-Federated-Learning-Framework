import { guard, notFound } from "@/lib/api";
import { getRound, parseId } from "@/lib/queries";

export async function GET(
  _: Request,
  ctx: RouteContext<"/api/v1/runs/[runId]/rounds/[roundId]">,
) {
  const denied = await guard();
  if (denied) return denied;
  const { runId, roundId } = await ctx.params;
  const id = parseId(roundId);
  const round = id === null ? null : await getRound(runId, id);
  return round ? Response.json(round) : notFound("round");
}
