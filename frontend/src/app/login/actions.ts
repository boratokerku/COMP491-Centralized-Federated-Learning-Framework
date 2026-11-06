"use server";

import { cookies, headers } from "next/headers";
import { redirect } from "next/navigation";
import { clearFailures, isLocked, recordFailure } from "@/lib/auth/rate-limit";
import { SESSION_COOKIE, SESSION_TTL_S, createSessionToken } from "@/lib/auth/session";
import { checkCredentials, loginConfigured } from "@/lib/auth/users";

export type LoginState = { error?: string };

// Only follow redirects that stay inside the internal area.
function safeNext(value: FormDataEntryValue | null): string {
  const next = typeof value === "string" ? value : "";
  return next.startsWith("/app") && !next.startsWith("//") ? next : "/app";
}

export async function login(_: LoginState, form: FormData): Promise<LoginState> {
  if (!loginConfigured()) {
    return { error: "Login is not configured on this server (SESSION_SECRET, WEB_USERS)." };
  }
  const username = String(form.get("username") ?? "").trim();
  const password = String(form.get("password") ?? "");
  const address = ((await headers()).get("x-forwarded-for") ?? "unknown").split(",")[0].trim();
  const key = `${address}|${username.toLowerCase()}`;

  if (isLocked(key)) return { error: "Too many failed attempts. Try again in a few minutes." };
  if (!username || !password || !(await checkCredentials(username, password))) {
    recordFailure(key);
    return { error: "Wrong username or password." };
  }
  clearFailures(key);
  (await cookies()).set(SESSION_COOKIE, createSessionToken(username), {
    httpOnly: true,
    sameSite: "lax",
    secure: (process.env.PUBLIC_URL ?? "").startsWith("https://"),
    path: "/",
    maxAge: SESSION_TTL_S,
  });
  redirect(safeNext(form.get("next")));
}

export async function logout(): Promise<void> {
  (await cookies()).delete(SESSION_COOKIE);
  redirect("/login");
}
