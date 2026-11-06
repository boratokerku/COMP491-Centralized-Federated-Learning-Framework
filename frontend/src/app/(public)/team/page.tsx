import type { Metadata } from "next";
import { Card, PageHeader } from "@/components/ui";
import { REPO_URL, advisors, team } from "@/content/site";

export const metadata: Metadata = { title: "Team" };

const initials = (name: string) =>
  name
    .split(" ")
    .map((p) => p[0])
    .join("");

export default function Team() {
  return (
    <div className="space-y-10">
      <PageHeader
        title="Team FedUp"
        description="We are FedUp, the team behind this COMP 491 capstone project at Koç University."
      >
        <a href={REPO_URL} className="rounded-lg border border-border px-4 py-2 text-sm hover:bg-surface">
          Source code on GitHub &rarr;
        </a>
      </PageHeader>
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Students</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {team.map((m) => (
            <Card key={m.name} className="flex gap-4">
              <span className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-accent-soft font-semibold text-accent">
                {initials(m.name)}
              </span>
              <div>
                <h3 className="font-semibold">{m.name}</h3>
                <p className="text-sm text-muted">{m.role}</p>
              </div>
            </Card>
          ))}
        </div>
      </section>
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Advisors</h2>
        <ul className="grid gap-4 sm:grid-cols-2">
          {advisors.map((a) => (
            <li key={a}>
              <Card>{a}</Card>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
