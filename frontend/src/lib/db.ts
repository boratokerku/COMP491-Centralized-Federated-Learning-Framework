import "server-only";
import { Pool, type QueryResultRow } from "pg";

// One pool per server process. DATABASE_URL uses the read-only role `cfl_web`
// (SELECT only, no access to clients.token_hash), so nothing here can write.
const globalForPool = globalThis as unknown as { cflPool?: Pool };

function pool(): Pool {
  if (!globalForPool.cflPool) {
    const connectionString = process.env.DATABASE_URL;
    if (!connectionString) throw new Error("DATABASE_URL is not set");
    globalForPool.cflPool = new Pool({ connectionString, max: 5 });
  }
  return globalForPool.cflPool;
}

export async function query<T extends QueryResultRow>(
  sql: string,
  params: unknown[] = [],
): Promise<T[]> {
  const result = await pool().query<T>(sql, params);
  return result.rows;
}

export async function queryOne<T extends QueryResultRow>(
  sql: string,
  params: unknown[] = [],
): Promise<T | null> {
  return (await query<T>(sql, params))[0] ?? null;
}
