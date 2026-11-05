export const REPO_URL =
  "https://github.com/boratokerku/COMP491-Centralized-Federated-Learning-Framework";

export const nav = [
  { href: "/", label: "Home" },
  { href: "/docs", label: "Documentation" },
  { href: "/results", label: "Results" },
  { href: "/team", label: "Team" },
];

export const team = [
  { name: "Bora Toker", role: "Aggregator, coordinator, gateway, shared client runtime, deployment" },
  { name: "Enes Tunç", role: "Bank client (tabular), API specification, independent-implementation client" },
  { name: "Hilal Aygül", role: "Website, Grafana dashboards, end-of-run report" },
  { name: "Merve Köybaşı", role: "Telecom client (text), conformance and database schema" },
];

export const advisors = ["Prof. Mehmet Emre Gürsoy", "Prof. Barış Akgün"];

// Weekly progress notes, newest first. Add one entry per week until the end of the semester.
export const updates: { week: string; date: string; items: string[] }[] = [
  {
    week: "Week 4",
    date: "2026-11-05",
    items: [
      "Shared client runtime merged: polling, retries, pending-update persistence and final-model verification.",
      "SMS Spam task spec and evaluation added for the text client.",
      "IID stratified partitioning with readable summaries and reproducibility tests for both datasets.",
      "Full server stack runs in containers; a 3-round simulation completes and every client verifies the final model.",
    ],
  },
  {
    week: "Week 3",
    date: "2026-11-04",
    items: [
      "Aggregator, round orchestration and coordinator resume merged.",
      "Gateway with client tokens, HTTP transport and containerization (Caddy, PostgreSQL, Grafana).",
    ],
  },
  {
    week: "Week 2",
    date: "2026-10-27",
    items: [
      "Database schema and conformance manifest specification.",
      "Data preparation: Bank Marketing and SMS Spam downloads, preprocessing and partition scripts.",
    ],
  },
  {
    week: "Week 1",
    date: "2026-10-26",
    items: [
      "Scope and requirements document, repository structure, architecture and sequence diagrams.",
    ],
  },
];
