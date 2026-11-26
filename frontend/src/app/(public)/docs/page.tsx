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

const alwaysOn = [
  ["Data stays local", "Clients train on their own data. They send model updates to the server, not raw records."],
  ["Encrypted transport", "Clients connect to the gateway over HTTPS only, and each one authenticates with its own token."],
  ["Checked model files", "Weights are exchanged as safetensors files with a SHA-256 checksum. Nothing received over the network is unpickled."],
];

const optional = [
  {
    title: "Clipping and noise",
    text: "Before uploading, a client limits the size (L2 norm) of its model change and adds Gaussian noise. The noise per client gets smaller as more clients are selected. In this mode, updates are weighted by a trust score instead of by sample count. The score uses participation, on-time delivery, reported sample counts and the stability of reported validation loss.",
  },
  {
    title: "Secure aggregation (SecAgg+)",
    text: "Builds on clipping and noise. Each client masks its update before uploading, and the masks are removed only from the combined sum. If some clients drop out, the round can still be unmasked as long as enough of the others stay to meet the recovery threshold. The design follows Bonawitz et al. (CCS 2017).",
  },
  {
    title: "Mutual TLS (mTLS)",
    text: "The client gateway also asks each client for a certificate signed by the project\u2019s own certificate authority. The name in the certificate must match the client that owns the token. The website keeps using normal HTTPS.",
  },
];

const limits = [
  "The added noise does not come with a formal differential privacy guarantee: no privacy budget (\u03b5) is calculated.",
  "Without secure aggregation, the server can see each client\u2019s clipped and noisy update.",
  "Client public keys used by secure aggregation are not signed. The protection assumes that the server passes them on unchanged.",
  "With secure aggregation, the server cannot sanity-check updates one by one. It checks the combined model, and a masked update that fails validation fails the whole round.",
];

const securityLinks = [
  { label: "Protocol in the API specification", href: blob("planning/api-spec.md") },
  { label: "Example privacy run config", href: blob("backend/configs/sim_tabular_privacy.yaml") },
  { label: "mTLS certificate script", href: blob("deploy/scripts/mtls_certs.sh") },
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

      <section className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-lg font-semibold">Security and privacy</h2>
          <p className="max-w-4xl text-sm text-muted">What protects a training run, and where that protection ends.</p>
        </div>

        <h3 className="text-xs font-medium uppercase tracking-wide text-muted">Always on</h3>
        <div className="grid gap-4 sm:grid-cols-3">
          {alwaysOn.map(([title, text]) => (
            <Card key={title}>
              <h4 className="font-semibold">{title}</h4>
              <p className="mt-1 text-sm text-muted">{text}</p>
            </Card>
          ))}
        </div>

        <h3 className="text-xs font-medium uppercase tracking-wide text-muted">Optional, off by default</h3>
        <div className="grid gap-4 sm:grid-cols-3">
          {optional.map((o) => (
            <Card key={o.title}>
              <h4 className="font-semibold">{o.title}</h4>
              <p className="mt-1 text-sm text-muted">{o.text}</p>
            </Card>
          ))}
        </div>

        <Card>
          <h3 className="font-semibold">Limits</h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-muted">
            {limits.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
          <ul className="mt-4 space-y-1 text-sm">
            {securityLinks.map((l) => (
              <li key={l.href}>
                <a href={l.href} className="text-accent hover:underline">
                  {l.label} &rarr;
                </a>
              </li>
            ))}
          </ul>
        </Card>
      </section>
    </div>
  );
}
