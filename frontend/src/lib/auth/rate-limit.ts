import "server-only";

// Failed logins per key (client address + username) in a sliding window. In memory: enough
// for a handful of accounts on one server, and it resets on restart.
const WINDOW_MS = 10 * 60 * 1000;
const MAX_FAILURES = 5;
const failures = new Map<string, number[]>();

function recent(key: string, now: number): number[] {
  const list = (failures.get(key) ?? []).filter((t) => now - t < WINDOW_MS);
  failures.set(key, list);
  return list;
}

export function isLocked(key: string, now = Date.now()): boolean {
  return recent(key, now).length >= MAX_FAILURES;
}

export function recordFailure(key: string, now = Date.now()): void {
  recent(key, now).push(now);
}

export function clearFailures(key: string): void {
  failures.delete(key);
}
