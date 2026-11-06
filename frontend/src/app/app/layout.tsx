import Link from "next/link";
import { logout } from "@/app/login/actions";
import { currentUser } from "@/lib/auth/guard";

const links = [
  { href: "/app/rounds", label: "Live round" },
  { href: "/app/runs", label: "Runs" },
  { href: "/app/dashboards", label: "Dashboards" },
  { href: "/app/updates", label: "Update inbox" },
  { href: "/app/logs", label: "Event log" },
  { href: "/app/clients", label: "Clients" },
  { href: "/app/reports", label: "Reports" },
];

export default async function AppLayout({ children }: LayoutProps<"/app">) {
  // Each page does its own requireUser() with its own path, so the login redirect can send
  // the visitor back to the page they asked for. Without a session the pages redirect and
  // render nothing of the area.
  const user = await currentUser();
  if (!user) return <>{children}</>;
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3 text-sm">
        <nav className="flex flex-wrap gap-4">
          {links.map((l) => (
            <Link key={l.href} href={l.href} className="text-muted hover:text-foreground">
              {l.label}
            </Link>
          ))}
        </nav>
        <form action={logout} className="flex items-center gap-2 text-muted">
          <span>{user}</span>
          <button type="submit" className="underline hover:text-foreground">
            Sign out
          </button>
        </form>
      </div>
      {children}
    </div>
  );
}
