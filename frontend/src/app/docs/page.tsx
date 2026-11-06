import type { Metadata } from "next";
import { REPO_URL } from "@/content/site";

export const metadata: Metadata = { title: "Documentation" };

const blob = (path: string) => `${REPO_URL}/blob/main/${path}`;

const docs = [
  {
    title: "Setup and run instructions",
    text: "Install the backend, prepare the data, start the server stack in containers and run a simulation.",
    links: [
      { label: "README", href: blob("README.md") },
      { label: "Deployment guide", href: blob("deploy/README.md") },
    ],
  },
  {
    title: "REST API reference",
    text: "Gateway endpoints for clients (register, conformance, task, model, update) and the read-only web API.",
    links: [{ label: "API specification", href: blob("planning/api-spec.md") }],
  },
  {
    title: "Conformance Manifest",
    text: "The manifest and report format a client must satisfy before it may take part in a run.",
    links: [
      { label: "Specification", href: blob("planning/conformance-manifest.md") },
      { label: "JSON schema", href: blob("planning/conformance-manifest.schema.json") },
      { label: "SMS Spam example", href: blob("planning/examples/sms_spam_manifest.json") },
      { label: "Bank Marketing example", href: blob("planning/examples/bank_marketing_manifest.json") },
    ],
  },
  {
    title: "Client integration guide",
    text: "What each client team implements (TaskSpec, round config, metric names) and how the protocol works.",
    links: [{ label: "Team interfaces and decisions", href: blob("planning/interfaces.md") }],
  },
  {
    title: "Project scope and data",
    text: "Requirements, feature boundaries, build order, and how the datasets are prepared and partitioned.",
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
      <h1 className="text-3xl font-bold">Documentation</h1>
      <p className="text-muted">
        The documents live in the{" "}
        <a href={REPO_URL} className="text-accent underline">
          GitHub repository
        </a>{" "}
        and are linked here.
      </p>
      {docs.map((d) => (
        <section key={d.title} className="space-y-2">
          <h2 className="text-xl font-semibold">{d.title}</h2>
          <p>{d.text}</p>
          <ul className="list-disc pl-6 text-sm">
            {d.links.map((l) => (
              <li key={l.href}>
                <a href={l.href} className="text-accent underline">
                  {l.label}
                </a>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
