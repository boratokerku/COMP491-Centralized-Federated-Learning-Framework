import "server-only";
import { apiGuard } from "./auth/guard";

export const notFound = (what: string) => Response.json({ error: `${what} not found` }, { status: 404 });
export const badRequest = (message: string) => Response.json({ error: message }, { status: 400 });

/** Every /api/v1 handler runs behind the session check; returns the 401 response or null. */
export const guard = apiGuard;
