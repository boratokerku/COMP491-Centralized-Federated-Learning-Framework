
import Link from "next/link"; 
import { logout } from "@/app/login/actions";
import { NavLink } from "@/components/nav-link";
import { NavPill } from "@/components/nav-pill";
import { Logo } from "@/components/ui";
import { currentUser } from "@/lib/auth/guard";

const tabs = [
  { href: "/app", label: "Overview", exact: true },
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
    <>
      <header className="sticky top-0 z-10 border-b border-border bg-background/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-6 pt-3">
          <div className="flex items-center gap-3">
            <Link href="/app" className="flex items-center gap-2 font-semibold">
              <Logo />
              FedUp
            </Link>
            <span className="rounded-full bg-accent-soft px-2.5 py-0.5 text-xs font-medium text-accent">
              Internal area
            </span>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <Link href="/" className="text-muted hover:text-foreground">
              Project site &#8599;
            </Link>
            <form action={logout} className="flex items-center gap-2">
              <span className="text-muted">{user}</span>
              <button
                type="submit"
                className="rounded-md border border-border px-2.5 py-1 hover:bg-surface"
              >
                Sign out
              </button>
            </form>
          </div>
        </div>
        <nav
          aria-label="Internal area"
          className="relative mx-auto flex max-w-6xl gap-1 overflow-x-auto px-4"
        >
          <NavPill underline />
          {tabs.map((t) => (
            <NavLink key={t.href} href={t.href} exact={t.exact} variant="tab">
              {t.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8">{children}</main>
    </>
  );
}
