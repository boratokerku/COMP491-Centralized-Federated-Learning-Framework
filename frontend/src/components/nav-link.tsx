"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { CSSProperties, ReactNode } from "react";

/** A menu link that shows when it is the current page. `exact` for "/" and "/app". */
export function NavLink({
  href,
  exact,
  variant = "top",
  hover,
  children,
}: {
  href: string;
  exact?: boolean;
  variant?: "top" | "tab";
  /** Colour on hover, for the top menu. */
  hover?: string;
  children: ReactNode;
}) {
  const pathname = usePathname();
  const active = exact ? pathname === href : pathname === href || pathname.startsWith(`${href}/`);
  const base =
    variant === "tab"
      ? "whitespace-nowrap border-b-2 px-3 py-3 text-sm transition-colors"
      : "rounded-md px-3 py-1.5 text-sm transition-colors";
  const state =
    variant === "tab"
      ? active
        ? "border-accent font-medium text-foreground"
        : "border-transparent text-muted hover:text-foreground"
      : active
        ? "bg-accent-soft font-medium text-accent"
        : "text-muted hover:bg-surface hover:text-foreground";
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={`${base} ${state} ${hover ? "nav-hover" : ""}`}
      style={hover ? ({ "--hover": hover } as CSSProperties) : undefined}
    >
      {children}
    </Link>
  );
}
