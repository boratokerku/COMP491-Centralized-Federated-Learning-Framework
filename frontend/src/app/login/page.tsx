import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { Card } from "@/components/ui";
import { currentUser } from "@/lib/auth/guard";
import { LoginForm } from "./login-form";

export const metadata: Metadata = { title: "Sign in" };

export default async function LoginPage({ searchParams }: PageProps<"/login">) {
  const next = (await searchParams).next;
  const target = typeof next === "string" ? next : "/app";
  if (await currentUser()) redirect(target.startsWith("/app") ? target : "/app");
  return (
    <Card className="space-y-6 p-6">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold">Internal area</h1>
        <p className="text-sm text-muted">Sign in with your team account to see runs, rounds and clients.</p>
      </div>
      <LoginForm next={target} />
    </Card>
  );
}
