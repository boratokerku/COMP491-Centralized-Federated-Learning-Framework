import "server-only";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { isUuid } from "./queries";

// Report files written by `cfl-report` to <REPORTS_DIR>/<run_id>/report/ (read-only mount).
export const REPORT_FILES = {
  summary: { file: "summary.json", type: "application/json" },
  markdown: { file: "report.md", type: "text/markdown; charset=utf-8" },
  "metrics.csv": { file: "metrics.csv", type: "text/csv; charset=utf-8" },
} as const;

export type ReportName = keyof typeof REPORT_FILES;
const PLOT_NAME = /^[a-z0-9_]{1,64}\.png$/;

function reportPath(runId: string, ...parts: string[]): string | null {
  const root = process.env.REPORTS_DIR;
  if (!root || !isUuid(runId)) return null;
  return path.join(root, runId, "report", ...parts);
}

async function read(file: string | null): Promise<Buffer | null> {
  if (!file) return null;
  try {
    return await readFile(file);
  } catch {
    return null;
  }
}

export const readReport = (runId: string, name: ReportName) =>
  read(reportPath(runId, REPORT_FILES[name].file));

export const readPlot = (runId: string, name: string) =>
  PLOT_NAME.test(name) ? read(reportPath(runId, "plots", name)) : Promise.resolve(null);

export async function hasReport(runId: string): Promise<boolean> {
  const file = reportPath(runId, REPORT_FILES.summary.file);
  if (!file) return false;
  try {
    return (await stat(file)).isFile();
  } catch {
    return false;
  }
}

export async function listPlots(runId: string): Promise<string[]> {
  const summary = await readReport(runId, "markdown");
  if (!summary) return [];
  return [...summary.toString().matchAll(/\(plots\/([a-z0-9_]+\.png)\)/g)].map((m) => m[1]);
}
