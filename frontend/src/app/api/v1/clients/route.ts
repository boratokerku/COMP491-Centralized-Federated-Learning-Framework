import { guard } from "@/lib/api";
import { listClients } from "@/lib/queries";

export async function GET() {
  const denied = await guard();
  if (denied) return denied;
  return Response.json({ clients: await listClients() });
}
