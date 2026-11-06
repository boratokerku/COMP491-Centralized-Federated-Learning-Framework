import { guard, notFound } from "@/lib/api";
import { getRun } from "@/lib/queries";
import { hasReport, listPlots } from "@/lib/reports";

export async function GET(_: Request, ctx: RouteContext<"/api/v1/runs/[runId]/reports">) {
  const denied = await guard();
  if (denied) return denied;
  const { runId } = await ctx.params;
  if (!(await getRun(runId))) return notFound("run");
  if (!(await hasReport(runId))) return notFound("report");
  const base = `/api/v1/runs/${runId}/reports`;
  return Response.json({
    run_id: runId,
    summary: `${base}/summary`,
    markdown: `${base}/markdown`,
    metrics_csv: `${base}/metrics.csv`,
    plots: (await listPlots(runId)).map((name) => ({ name, url: `${base}/plots/${name}` })),
  });
}
