import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { REPO_URL, nav } from "@/content/site";

export const metadata: Metadata = {
  title: {
    default: "Centralized Federated Learning Framework",
    template: "%s | CFL Framework",
  },
  description:
    "A cross-silo federated learning framework with conformance checking, fault tolerance and observability. COMP 491 capstone, Koç University.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">
        <header className="border-b border-border">
          <div className="mx-auto flex max-w-4xl flex-wrap items-center justify-between gap-3 px-6 py-4">
            <Link href="/" className="font-semibold">
              CFL Framework
            </Link>
            <nav className="flex gap-5 text-sm">
              {nav.map((item) => (
                <Link key={item.href} href={item.href} className="text-muted hover:text-foreground">
                  {item.label}
                </Link>
              ))}
              <a href={REPO_URL} className="text-muted hover:text-foreground">
                GitHub
              </a>
            </nav>
          </div>
        </header>
        <main className="mx-auto w-full max-w-4xl flex-1 px-6 py-10">{children}</main>
        <footer className="border-t border-border">
          <div className="mx-auto max-w-4xl px-6 py-6 text-sm text-muted">
            COMP 491 capstone project, Koç University.
          </div>
        </footer>
      </body>
    </html>
  );
}
