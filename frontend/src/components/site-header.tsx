import Image from "next/image";
import Link from "next/link";
import { REPO_URL, nav } from "@/content/site";
import { NavLink } from "./nav-link";
import { Logo } from "./ui";
import githubLogo from "../../assets/github_logo.png";

export function SiteHeader() {
  return (
    <header className="top-0 z-10 border-b md:sticky border-border bg-background/90 backdrop-blur">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-x-6 gap-y-2 px-6 py-3">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <Logo />
          FedUp
        </Link>
        <nav aria-label="Main" className="flex flex-wrap items-center gap-1">
          {nav.map((item) => (
            <NavLink key={item.href} href={item.href} exact={item.href === "/"} hover={item.hover}>
              {item.label}
            </NavLink>
          ))}
          <a
            href={REPO_URL}
            aria-label="GitHub"
            className="ml-1 rounded-md bg-white px-1 transition-opacity hover:opacity-90"
          >
            <Image src={githubLogo} alt="" className="h-8 w-auto" />
          </a>
          {/* A signed-in visitor is sent on to /app by the login page. */}
          <Link
            href="/login"
            prefetch={false}
            className="ml-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-background hover:opacity-90"
          >
            Admin login
          </Link>
        </nav>
      </div>
    </header>
  );
}
