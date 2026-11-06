import type { Metadata } from "next";
import { Card, PageHeader } from "@/components/ui";
import { REPO_URL } from "@/content/site";

export const metadata: Metadata = { title: "Documentation" };

const blob = (path: string) => `${REPO_URL}/blob/main/${path}`;

const docs = [
  {
    title: "Setup and run",
    text: "Install the backend, prepare the data, start the server in containers and run a simulation.",
    links: [
      { label: "README", href: blob("README.md") },
      { label: "Deployment guide", href: blob("deploy/README.md") },
    ],
  },
  {
    title: "REST API",
    text: "Gateway endpoints for clients (register, conformance, task, model, update) and the read-only web API.",
    links: [{ label: "API specification", href: blob("planning/api-spec.md") }],
  },
  {
    title: "Conformance manifest",
    text: "The manifest and report format a client must satisfy before it may take part in a run.",
    links: [
      { label: "Specification", href: blob("planning/conformance-manifest.md") },
      { label: "JSON schema", href: blob("planning/conformance-manifest.schema.json") },
      { label: "SMS Spam example", href: blob("planning/examples/sms_spam_manifest.json") },
      { label: "Bank Marketing example", href: blob("planning/examples/bank_marketing_manifest.json") },
    ],
  },
  {
    title: "Client integration",
    text: "What each client team implements (TaskSpec, round config, metric names) and how the protocol works.",
    links: [{ label: "Team interfaces and decisions", href: blob("planning/interfaces.md") }],
  },
  {
    title: "Scope and data",
    text: "Requirements, feature boundaries, build order, and how the data sets are prepared and partitioned.",
    links: [
      { label: "Scope", href: blob("planning/scope.md") },
      { label: "Data", href: blob("planning/data.md") },
      { label: "Database schema", href: blob("planning/db-schema.md") },
    ],
  },
];

export default function Docs() {
  return (
    <div className="space-y-8">
      <PageHeader
        title="Documentation"
        description="The documents live in the GitHub repository. Each card below links to the ones you need."
      />
      <div className="grid gap-4 sm:grid-cols-2">
        {docs.map((d) => (
          <Card key={d.title} className="flex flex-col">
            <h2 className="font-semibold">{d.title}</h2>
            <p className="mt-1 flex-1 text-sm text-muted">{d.text}</p>
            <ul className="mt-4 space-y-1 text-sm">
              {d.links.map((l) => (
                <li key={l.href}>
                  <a href={l.href} className="text-accent hover:underline">
                    {l.label} &rarr;
                  </a>
                </li>
              ))}
            </ul>
          </Card>
        ))}
      </div>
    </div>
  );
}
