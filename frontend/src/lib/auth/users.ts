import "server-only";
import { readFileSync } from "node:fs";
import bcrypt from "bcryptjs";

// Accounts are `name:bcrypt-hash`, one per line or comma separated, from WEB_USERS or from
// the file named by WEB_USERS_FILE (a Docker secret works). Passwords are never stored.
function loadUsers(): Map<string, string> {
  const parts: string[] = [];
  if (process.env.WEB_USERS) parts.push(process.env.WEB_USERS);
  if (process.env.WEB_USERS_FILE) parts.push(readFileSync(process.env.WEB_USERS_FILE, "utf8"));
  const users = new Map<string, string>();
  for (const entry of parts.join("\n").split(/[\n,]/)) {
    const line = entry.trim();
    const colon = line.indexOf(":");
    if (!line || line.startsWith("#") || colon < 1) continue;
    users.set(line.slice(0, colon), line.slice(colon + 1));
  }
  return users;
}

// Compared against when the user does not exist, so timing does not reveal valid names.
const DUMMY_HASH = bcrypt.hashSync("not-a-real-password", 10);

export async function checkCredentials(username: string, password: string): Promise<boolean> {
  const hash = loadUsers().get(username);
  const ok = await bcrypt.compare(password, hash ?? DUMMY_HASH);
  return hash !== undefined && ok;
}

export function loginConfigured(): boolean {
  return loadUsers().size > 0 && (process.env.SESSION_SECRET ?? "").length >= 32;
}
