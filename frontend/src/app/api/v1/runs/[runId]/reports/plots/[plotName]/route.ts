import { guard, notFound } from "@/lib/api";
import { readPlot } from "@/lib/reports";

export async function GET(
  _: Request,
  ctx: RouteContext<"/api/v1/runs/[runId]/reports/plots/[plotName]">,
) {
  const denied = await guard();
  if (denied) return denied;
  const { runId, plotName } = await ctx.params;
  const body = await readPlot(runId, plotName);
  if (!body) return notFound("plot");
  return new Response(new Uint8Array(body), { headers: { "Content-Type": "image/png" } });
}
