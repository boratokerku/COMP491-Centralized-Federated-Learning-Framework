import { NextResponse } from "next/server";
import { currentUser } from "@/lib/auth/guard";

// Caddy's forward_auth asks this before it lets a request through to Grafana: 200 with the
// user name when the session cookie is valid. Otherwise 401, or with ?redirect=1 a redirect
// to the login form (Caddy hands a non-2xx answer back to the visitor as it is).
export async function GET(request: Request) {
  const user = await currentUser();
  if (!user) {
    if (new URL(request.url).searchParams.has("redirect")) {
      return NextResponse.redirect(new URL("/login?next=/app/dashboards", publicOrigin(request)));
    }
    return new Response(null, { status: 401 });
  }
  return new Response(null, { status: 200, headers: { "X-Auth-User": user } });
}

// Behind Caddy the request URL says http://web:3000; send the visitor to the public address.
function publicOrigin(request: Request): string {
  return process.env.PUBLIC_URL ?? new URL(request.url).origin;
}
