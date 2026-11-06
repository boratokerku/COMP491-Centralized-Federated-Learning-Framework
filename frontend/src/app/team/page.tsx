import type { Metadata } from "next";
import { REPO_URL, advisors, team } from "@/content/site";

export const metadata: Metadata = { title: "Team" };

export default function Team() {
  return (
    <div className="space-y-8">
      <h1 className="text-3xl font-bold">Team and About</h1>
      <p>
        This is a COMP 491 capstone project at Koç University. The source code is on{" "}
        <a href={REPO_URL} className="text-accent underline">
          GitHub
        </a>
        .
      </p>
      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Team</h2>
        <ul className="space-y-2">
          {team.map((m) => (
            <li key={m.name}>
              <span className="font-medium">{m.name}</span>
              <span className="block text-sm text-muted">{m.role}</span>
            </li>
          ))}
        </ul>
      </section>
      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Advisors</h2>
        <ul className="list-disc pl-6">
          {advisors.map((a) => (
            <li key={a}>{a}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
