import { badRequest, guard } from "@/lib/api";
import { RUN_STATUSES, listRuns, page } from "@/lib/queries";

export async function GET(request: Request) {
  const denied = await guard();
  if (denied) return denied;
  const params = new URL(request.url).searchParams;
  const status = params.get("status") ?? undefined;
  if (status && !(RUN_STATUSES as readonly string[]).includes(status)) {
    return badRequest(`status must be one of ${RUN_STATUSES.join(", ")}`);
  }
  return Response.json(
    await listRuns({ status, ...page(params.get("limit"), params.get("offset")) }),
  );
}
