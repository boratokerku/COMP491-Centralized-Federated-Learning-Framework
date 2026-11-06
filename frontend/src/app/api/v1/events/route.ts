import { badRequest, guard } from "@/lib/api";
import { isUuid, listEvents, page, parseId } from "@/lib/queries";

export async function GET(request: Request) {
  const denied = await guard();
  if (denied) return denied;
  const params = new URL(request.url).searchParams;
  const runId = params.get("run_id") ?? undefined;
  if (runId && !isUuid(runId)) return badRequest("run_id must be a UUID");
  const roundParam = params.get("round_id");
  const roundId = roundParam === null ? undefined : parseId(roundParam);
  if (roundId === null) return badRequest("round_id must be a positive integer");
  return Response.json(
    await listEvents({
      run_id: runId,
      round_id: roundId,
      client_id: params.get("client_id") ?? undefined,
      event_type: params.get("event_type") ?? undefined,
      status: params.get("status") ?? undefined,
      ...page(params.get("limit"), params.get("offset")),
    }),
  );
}
