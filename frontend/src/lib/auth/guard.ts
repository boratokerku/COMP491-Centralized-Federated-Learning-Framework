import "server-only";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { SESSION_COOKIE, verifySessionToken } from "./session";

export async function currentUser(): Promise<string | null> {
  return verifySessionToken((await cookies()).get(SESSION_COOKIE)?.value);
}

/** For pages: send visitors without a session to the login form. */
export async function requireUser(next = "/app"): Promise<string> {
  const user = await currentUser();
  if (!user) redirect(`/login?next=${encodeURIComponent(next)}`);
  return user;
}

/** For route handlers: a JSON 401 instead of a redirect. */
export async function apiGuard(): Promise<Response | null> {
  if (await currentUser()) return null;
  return Response.json({ error: "unauthorized" }, { status: 401 });
}
