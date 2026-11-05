import { guard, notFound } from "@/lib/api";
import { getClient } from "@/lib/queries";

export async function GET(_: Request, ctx: RouteContext<"/api/v1/clients/[clientId]">) {
  const denied = await guard();
  if (denied) return denied;
  const client = await getClient(decodeURIComponent((await ctx.params).clientId));
  return client ? Response.json(client) : notFound("client");
}
