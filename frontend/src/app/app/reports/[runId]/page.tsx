import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { requireUser } from "@/lib/auth/guard";
import { getRun } from "@/lib/queries";
import { readReport } from "@/lib/reports";

export const metadata: Metadata = { title: "Report" };

export default async function ReportView({ params }: PageProps<"/app/reports/[runId]">) {
  const { runId } = await params;
  await requireUser(`/app/reports/${runId}`);
  const run = await getRun(runId);
  const markdown = run ? await readReport(runId, "markdown") : null;
  if (!run || !markdown) notFound();

  const base = `/api/v1/runs/${runId}/reports`;
  return (
    <div className="space-y-6">
      <p className="flex flex-wrap gap-4 text-sm">
        <Link href="/app/reports" className="text-accent underline">
          All reports
        </Link>
        <a href={`${base}/markdown`} className="text-accent underline" download={`run_${runId}_report.md`}>
          report.md
        </a>
        <a href={`${base}/summary`} className="text-accent underline" download={`run_${runId}_summary.json`}>
          summary.json
        </a>
        <a href={`${base}/metrics.csv`} className="text-accent underline" download={`run_${runId}_metrics.csv`}>
          metrics.csv
        </a>
      </p>
      <article className="space-y-3 text-sm [&_code]:font-mono [&_code]:text-xs [&_h1]:text-2xl [&_h1]:font-bold [&_h2]:mt-6 [&_h2]:text-lg [&_h2]:font-semibold [&_h3]:font-medium [&_pre]:overflow-x-auto [&_pre]:rounded [&_pre]:border [&_pre]:border-border [&_pre]:p-3 [&_table]:block [&_table]:overflow-x-auto [&_td]:border-b [&_td]:border-border [&_td]:px-2 [&_td]:py-1 [&_th]:px-2 [&_th]:py-1 [&_th]:text-left">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          urlTransform={(url) => (url.startsWith("plots/") ? `${base}/${url}` : url)}
          components={{
            // eslint-disable-next-line @next/next/no-img-element
            img: ({ src, alt }) => <img src={typeof src === "string" ? src : ""} alt={alt ?? ""} className="max-w-xl" />,
          }}
        >
          {markdown.toString()}
        </ReactMarkdown>
      </article>
    </div>
  );
}
