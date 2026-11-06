import Link from "next/link";
import { Badge } from "@/components/ui";
import { shortId } from "@/lib/format";
import { listRuns } from "@/lib/queries";

/** Links to switch between runs; `param` is the query parameter the page reads. */
export async function RunPicker({
  base,
  param,
  current,
}: {
  base: string;
  param: string;
  current: string;
}) {
  const { runs } = await listRuns({ limit: 8, offset: 0 });
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <span className="text-muted">Run:</span>
      {runs.map((r) => (
        <Link
          key={r.run_id}
          href={`${base}?${param}=${r.run_id}`}
          className={`flex items-center gap-1 rounded border px-2 py-1 ${
            r.run_id === current ? "border-accent" : "border-border"
          }`}
        >
          {r.name} <span className="font-mono text-xs text-muted">{shortId(r.run_id)}</span>
          <Badge value={r.status} />
        </Link>
      ))}
    </div>
  );
}
