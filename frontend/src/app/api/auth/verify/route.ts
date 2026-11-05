import { currentUser } from "@/lib/auth/guard";

// Caddy's forward_auth asks this before it lets a request through to Grafana: 200 with the
// user name when the session cookie is valid, 401 otherwise.
export async function GET() {
  const user = await currentUser();
  if (!user) return new Response(null, { status: 401 });
  return new Response(null, { status: 200, headers: { "X-Auth-User": user } });
}
