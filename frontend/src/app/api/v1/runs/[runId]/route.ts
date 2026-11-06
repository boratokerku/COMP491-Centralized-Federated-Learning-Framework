import { guard, notFound } from "@/lib/api";
import { getRun } from "@/lib/queries";

export async function GET(_: Request, ctx: RouteContext<"/api/v1/runs/[runId]">) {
  const denied = await guard();
  if (denied) return denied;
  const run = await getRun((await ctx.params).runId);
  return run ? Response.json(run) : notFound("run");
}
