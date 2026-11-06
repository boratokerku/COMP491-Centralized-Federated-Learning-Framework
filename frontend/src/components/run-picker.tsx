import Link from "next/link";
import { Badge } from "@/components/ui";
import { shortId } from "@/lib/format";
import { listRuns } from "@/lib/queries";

/** Pills to switch between runs; `param` is the query parameter the page reads. */
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
      <span className="text-muted">Run</span>
      {runs.map((r) => {
        const active = r.run_id === current;
        return (
          <Link
            key={r.run_id}
            href={`${base}?${param}=${r.run_id}`}
            aria-current={active ? "true" : undefined}
            className={`flex items-center gap-2 rounded-full border px-3 py-1 ${
              active ? "border-accent bg-accent-soft" : "border-border hover:bg-surface"
            }`}
          >
            {r.name} <span className="font-mono text-xs text-muted">{shortId(r.run_id)}</span>
            <Badge value={r.status} />
          </Link>
        );
      })}
    </div>
  );
}
