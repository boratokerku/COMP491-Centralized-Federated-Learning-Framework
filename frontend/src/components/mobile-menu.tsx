"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type CSSProperties } from "react";
import { REPO_URL, nav } from "@/content/site";
import { GitHubIcon } from "./ui";

/** The menu on phones and tablets: a button that drops the links down below the header. */
export function MobileMenu() {
  const pathname = usePathname();
  // Remembering the page the menu was opened on closes it by itself after a navigation.
  const [openOn, setOpenOn] = useState<string | null>(null);
  const open = openOn === pathname;

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpenOn(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <div className="lg:hidden">
      <button
        type="button"
        aria-expanded={open}
        aria-controls="mobile-menu"
        aria-label={open ? "Close menu" : "Open menu"}
        onClick={() => setOpenOn(open ? null : pathname)}
        className="menu-button grid h-10 w-10 place-items-center rounded-lg border border-border bg-surface"
      >
        <span aria-hidden className="menu-icon" data-open={open || undefined}>
          <span />
          <span />
          <span />
        </span>
      </button>

      <div id="mobile-menu" className="mobile-menu absolute inset-x-0 top-full" data-open={open || undefined}>
        <div className="overflow-hidden">
          <nav aria-label="Main" className="wrap border-b border-border bg-background pt-3 pb-6 shadow-[0_24px_48px_-12px_rgb(0_0_0/0.8)]">
            <ul>
              {nav.map((item, i) => {
                const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
                return (
                  <li key={item.href} className="mobile-menu-item" style={{ "--n": i } as CSSProperties}>
                    <Link
                      href={item.href}
                      aria-current={active ? "page" : undefined}
                      // A link to another page closes the menu through the page change; the link
                      // to this page goes nowhere, so it closes the menu itself.
                      onClick={active ? () => setOpenOn(null) : undefined}
                      className="flex items-center gap-3 border-b border-border/60 py-3.5 text-xl font-medium transition-colors"
                      style={{ color: active ? item.hover : undefined }}
                    >
                      <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: item.hover }} />
                      {item.label}
                    </Link>
                  </li>
                );
              })}
            </ul>
            <div className="mobile-menu-item mt-5 flex gap-3" style={{ "--n": nav.length } as CSSProperties}>
              <a
                href={REPO_URL}
                className="inline-flex h-11 flex-1 items-center justify-center gap-2 rounded-lg bg-foreground text-sm font-medium text-background"
              >
                <GitHubIcon className="h-4 w-4" />
                GitHub
              </a>
              <Link
                href="/login"
                prefetch={false}
                className="login-button inline-flex h-11 flex-1 items-center justify-center gap-1.5 rounded-lg text-sm font-medium"
              >
                Admin login
              </Link>
            </div>
          </nav>
        </div>
      </div>
    </div>
  );
}
