import { guard, notFound } from "@/lib/api";
import { REPORT_FILES, readReport } from "@/lib/reports";

export async function GET(
  _: Request,
  ctx: RouteContext<"/api/v1/runs/[runId]/reports/markdown">,
) {
  const denied = await guard();
  if (denied) return denied;
  const body = await readReport((await ctx.params).runId, "markdown");
  if (!body) return notFound("report file");
  return new Response(new Uint8Array(body), {
    headers: { "Content-Type": REPORT_FILES["markdown"].type },
  });
}
