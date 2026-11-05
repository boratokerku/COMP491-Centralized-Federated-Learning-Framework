import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { currentUser } from "@/lib/auth/guard";
import { LoginForm } from "./login-form";

export const metadata: Metadata = { title: "Sign in" };

export default async function LoginPage({ searchParams }: PageProps<"/login">) {
  const next = (await searchParams).next;
  const target = typeof next === "string" ? next : "/app";
  if (await currentUser()) redirect(target.startsWith("/app") ? target : "/app");
  return (
    <div className="mx-auto max-w-sm space-y-6">
      <h1 className="text-2xl font-bold">Internal area</h1>
      <p className="text-sm text-muted">For the project team and advisors.</p>
      <LoginForm next={target} />
    </div>
  );
}
