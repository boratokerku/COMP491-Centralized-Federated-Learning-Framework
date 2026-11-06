"use client";

import { useActionState } from "react";
import { login, type LoginState } from "./actions";

export function LoginForm({ next }: { next: string }) {
  const [state, action, pending] = useActionState<LoginState, FormData>(login, {});
  const field = "w-full rounded-lg border border-border bg-background px-3 py-2";
  return (
    <form action={action} className="space-y-4">
      <input type="hidden" name="next" value={next} />
      <label className="block space-y-1 text-sm">
        <span>Username</span>
        <input name="username" autoComplete="username" required className={field} />
      </label>
      <label className="block space-y-1 text-sm">
        <span>Password</span>
        <input
          name="password"
          type="password"
          autoComplete="current-password"
          required
          className={field}
        />
      </label>
      {state.error && (
        <p role="alert" className="text-sm text-danger">
          {state.error}
        </p>
      )}
      <button
        type="submit"
        disabled={pending}
        className="w-full rounded-lg bg-accent px-4 py-2 text-sm font-medium text-background hover:opacity-90 disabled:opacity-60"
      >
        {pending ? "Signing in..." : "Sign in"}
      </button>
    </form>
  );
}
