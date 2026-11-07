import Link from "next/link";
import { REPO_URL, nav } from "@/content/site";
import { MobileMenu } from "./mobile-menu";
import { NavLink } from "./nav-link";
import { NavPill } from "./nav-pill";
import { GitHubIcon, Logo } from "./ui";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-20 border-b border-border bg-background/90 backdrop-blur">
      <div className="wrap flex items-center justify-between gap-x-6 py-3">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <Logo />
          FedUp
        </Link>
        <nav aria-label="Main" className="relative isolate hidden items-center gap-1 lg:flex">
          <NavPill />
          {nav.map((item) => (
            <NavLink key={item.href} href={item.href} exact={item.href === "/"} hover={item.hover}>
              {item.label}
            </NavLink>
          ))}
          <a
            href={REPO_URL}
            aria-label="GitHub"
            className="ml-1 inline-flex h-8 items-center gap-1.5 rounded-md bg-foreground px-2.5 text-sm font-medium text-background transition-opacity hover:opacity-90"
          >
            <GitHubIcon className="h-4 w-4" />
            GitHub
          </a>
          {/* A signed-in visitor is sent on to /app by the login page. */}
          <Link
            href="/login"
            prefetch={false}
            className="login-button ml-1 inline-flex h-8 items-center gap-1 rounded-md px-2.5 text-sm font-medium"
          >
            <svg viewBox="0 0 16 16" aria-hidden className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={1.6}>
              <rect x="3" y="7" width="10" height="7" rx="1.5" />
              <path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" />
            </svg>
            Admin login
          </Link>
        </nav>
        <MobileMenu />
      </div>
    </header>
  );
}
