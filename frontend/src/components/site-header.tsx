import Link from "next/link";
import { REPO_URL, nav } from "@/content/site";
import { NavLink } from "./nav-link";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-10 border-b border-border bg-background/90 backdrop-blur">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-x-6 gap-y-2 px-6 py-3">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <span className="grid h-7 w-7 place-items-center rounded-md bg-accent text-xs font-bold text-white">
            FL
          </span>
          CFL Framework
        </Link>
        <nav aria-label="Main" className="flex flex-wrap items-center gap-1">
          {nav.map((item) => (
            <NavLink key={item.href} href={item.href} exact={item.href === "/"}>
              {item.label}
            </NavLink>
          ))}
          <a
            href={REPO_URL}
            className="ml-1 rounded-md border border-border px-3 py-1.5 text-sm text-muted hover:text-foreground"
          >
            GitHub
          </a>
          {/* A signed-in visitor is sent on to /app by the login page. */}
          <Link
            href="/login"
            prefetch={false}
            className="ml-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:opacity-90"
          >
            Admin login
          </Link>
        </nav>
      </div>
    </header>
  );
}
