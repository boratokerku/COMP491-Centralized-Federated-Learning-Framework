import "server-only";
import { createHmac, timingSafeEqual } from "node:crypto";

export const SESSION_COOKIE = "cfl_session";
export const SESSION_TTL_S = 8 * 60 * 60;

function secret(): string {
  const value = process.env.SESSION_SECRET;
  if (!value || value.length < 32) {
    throw new Error("SESSION_SECRET must be set to at least 32 characters");
  }
  return value;
}

function sign(payload: string): string {
  return createHmac("sha256", secret()).update(payload).digest("base64url");
}

/** A signed, expiring token: base64url(JSON{u, exp}).signature. Holds no secrets. */
export function createSessionToken(username: string, now = Date.now()): string {
  const payload = Buffer.from(
    JSON.stringify({ u: username, exp: Math.floor(now / 1000) + SESSION_TTL_S }),
  ).toString("base64url");
  return `${payload}.${sign(payload)}`;
}

export function verifySessionToken(token: string | undefined, now = Date.now()): string | null {
  if (!token) return null;
  const [payload, signature, extra] = token.split(".");
  if (!payload || !signature || extra !== undefined) return null;
  const expected = Buffer.from(sign(payload));
  const given = Buffer.from(signature);
  if (expected.length !== given.length || !timingSafeEqual(expected, given)) return null;
  try {
    const { u, exp } = JSON.parse(Buffer.from(payload, "base64url").toString());
    if (typeof u !== "string" || typeof exp !== "number" || exp * 1000 < now) return null;
    return u;
  } catch {
    return null;
  }
}
