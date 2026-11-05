import "server-only";
import { query, queryOne } from "./db";

// Read-only queries behind both the pages and /api/v1 (planning/api-spec.md section 4).
// Column lists are explicit: `clients.token_hash` is never selected.

export const RUN_STATUSES = ["PENDING", "RUNNING", "COMPLETED", "FAILED", "ABORTED"] as const;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export const isUuid = (value: string) => UUID.test(value);
export const parseId = (value: string) => (/^\d{1,9}$/.test(value) ? Number(value) : null);

export function page(limit: string | null, offset: string | null) {
  const l = Number(limit ?? 50);
  const o = Number(offset ?? 0);
  return {
    limit: Number.isInteger(l) && l > 0 ? Math.min(l, 500) : 50,
    offset: Number.isInteger(o) && o > 0 ? o : 0,
  };
}

const RUN_LIST = `run_id, name, status, strategy, task_name, seed::float8 AS seed, model_version,
  manifest_version, started_at, completed_at, created_at`;

export type RunSummary = {
  run_id: string;
  name: string;
  status: string;
  strategy: string;
  task_name: string;
  seed: number;
  model_version: number;
  manifest_version: string;
  started_at: Date | null;
  completed_at: Date | null;
  created_at: Date;
};

export async function listRuns(opts: { status?: string; limit: number; offset: number }) {
  const where = opts.status ? "WHERE status = $1" : "";
  const filter = opts.status ? [opts.status] : [];
  const [count] = await query<{ n: number }>(`SELECT count(*)::int AS n FROM runs ${where}`, filter);
  const runs = await query<RunSummary>(
    `SELECT ${RUN_LIST} FROM runs ${where}
     ORDER BY created_at DESC LIMIT $${filter.length + 1} OFFSET $${filter.length + 2}`,
    [...filter, opts.limit, opts.offset],
  );
  return { total: count.n, runs };
}

export type RunDetail = RunSummary & {
  config: Record<string, unknown>;
  config_sha256: string;
  protocol_version: string;
  manifest_sha256: string;
  preprocessing_version: string;
  model_architecture: string;
  model_architecture_version: string;
  model_fingerprint: string;
  model_path: string | null;
  model_sha256: string | null;
  failure_reason: string | null;
  final_model_path: string | null;
  final_model_sha256: string | null;
  final_model_format: string | null;
  final_model_bytes: number | null;
};

export async function getRun(runId: string): Promise<RunDetail | null> {
  if (!isUuid(runId)) return null;
  return queryOne<RunDetail>(
    `SELECT ${RUN_LIST}, config, config_sha256, protocol_version, manifest_sha256,
       preprocessing_version, model_architecture, model_architecture_version, model_fingerprint,
       model_path, model_sha256, failure_reason, final_model_path, final_model_sha256,
       final_model_format, final_model_bytes::float8 AS final_model_bytes
     FROM runs WHERE run_id = $1`,
    [runId],
  );
}

/** The run to show by default: a running one, otherwise the most recent. */
export async function defaultRunId(): Promise<string | null> {
  const row = await queryOne<{ run_id: string }>(
    `SELECT run_id FROM runs
     ORDER BY (status = 'RUNNING') DESC, created_at DESC LIMIT 1`,
  );
  return row?.run_id ?? null;
}

export type RoundRow = {
  round_id: number;
  state: string;
  input_model_version: number;
  output_model_version: number | null;
  min_participants: number;
  selected_count: number;
  accepted_count: number;
  opened_at: Date;
  deadline_at: Date;
  closed_at: Date | null;
  failure_reason: string | null;
};

export async function listRounds(runId: string): Promise<RoundRow[]> {
  if (!isUuid(runId)) return [];
  return query<RoundRow>(
    `SELECT r.round_id, r.state, r.input_model_version, r.output_model_version,
       r.min_participants, r.opened_at, r.deadline_at, r.closed_at, r.failure_reason,
       count(p.client_id)::int AS selected_count,
       (count(p.client_id) FILTER (WHERE p.status = 'UPDATE_ACCEPTED'))::int AS accepted_count
     FROM rounds r
     LEFT JOIN round_participants p USING (run_id, round_id)
     WHERE r.run_id = $1
     GROUP BY r.run_id, r.round_id ORDER BY r.round_id`,
    [runId],
  );
}

export type Participant = {
  client_id: string;
  status: string;
  reason_code: string | null;
  selected_at: Date;
  finished_at: Date | null;
};

export type MetricPoint = {
  metric_name: string;
  value: number;
  unit: string;
  client_id: string | null;
  message_type: string | null;
};

export async function getRound(runId: string, roundId: number) {
  if (!isUuid(runId)) return null;
  const round = (await listRounds(runId)).find((r) => r.round_id === roundId);
  if (!round) return null;
  const participants = await query<Participant>(
    `SELECT client_id, status, reason_code, selected_at, finished_at
     FROM round_participants WHERE run_id = $1 AND round_id = $2 ORDER BY client_id`,
    [runId, roundId],
  );
  const metrics = await query<MetricPoint>(
    `SELECT metric_name, value, unit, client_id, message_type
     FROM metrics WHERE run_id = $1 AND round_id = $2 AND client_id IS NULL
     ORDER BY metric_name`,
    [runId, roundId],
  );
  return { ...round, run_id: runId, participants, metrics };
}

export type Conformance = {
  manifest_version: string;
  status: string;
  reason_code: string | null;
  checked_at: Date;
};

const CLIENT_COLUMNS = `c.client_id, c.status, c.client_version, c.protocol_version,
  c.last_seen_at, c.registered_at`;

export async function listClients() {
  const rows = await query<{
    client_id: string;
    status: string;
    client_version: string | null;
    protocol_version: string | null;
    last_seen_at: Date | null;
    registered_at: Date | null;
    cf_manifest_version: string | null;
    cf_status: string | null;
    cf_reason_code: string | null;
    cf_checked_at: Date | null;
  }>(
    `SELECT ${CLIENT_COLUMNS}, l.manifest_version AS cf_manifest_version,
       l.status AS cf_status, l.reason_code AS cf_reason_code, l.checked_at AS cf_checked_at
     FROM clients c
     LEFT JOIN LATERAL (
       SELECT manifest_version, status, reason_code, checked_at FROM conformance_reports
       WHERE client_id = c.client_id ORDER BY checked_at DESC LIMIT 1
     ) l ON true
     ORDER BY c.client_id`,
  );
  return rows.map((r) => ({
    client_id: r.client_id,
    status: r.status,
    client_version: r.client_version,
    protocol_version: r.protocol_version,
    last_seen_at: r.last_seen_at,
    registered_at: r.registered_at,
    latest_conformance:
      r.cf_status && r.cf_manifest_version && r.cf_checked_at
        ? {
            manifest_version: r.cf_manifest_version,
            status: r.cf_status,
            reason_code: r.cf_reason_code,
            checked_at: r.cf_checked_at,
          }
        : null,
  }));
}

export async function getClient(clientId: string) {
  const client = await queryOne<{
    client_id: string;
    status: string;
    client_version: string | null;
    protocol_version: string | null;
    created_at: Date;
    registered_at: Date | null;
    last_seen_at: Date | null;
  }>(
    `SELECT client_id, status, client_version, protocol_version, created_at, registered_at,
       last_seen_at FROM clients c WHERE client_id = $1`,
    [clientId],
  );
  if (!client) return null;
  const conformance = await query<
    Conformance & { report_id: string; client_version: string; report_payload: unknown }
  >(
    `SELECT report_id, manifest_version, client_version, status, reason_code, report_payload,
       checked_at FROM conformance_reports WHERE client_id = $1 ORDER BY checked_at DESC`,
    [clientId],
  );
  const participation = await query<{
    run_id: string;
    round_id: number;
    status: string;
    reason_code: string | null;
    finished_at: Date | null;
  }>(
    `SELECT run_id, round_id, status, reason_code, finished_at FROM round_participants
     WHERE client_id = $1 ORDER BY selected_at DESC LIMIT 200`,
    [clientId],
  );
  const metrics = await query<MetricPoint & { run_id: string; round_id: number }>(
    `SELECT run_id, round_id, metric_name, value, unit, client_id, message_type
     FROM metrics WHERE client_id = $1 ORDER BY recorded_at DESC LIMIT 500`,
    [clientId],
  );
  return { ...client, conformance, participation, metrics };
}

export type UpdateRow = {
  update_id: string;
  client_id: string;
  model_version: number;
  num_samples: number;
  checksum_sha256: string;
  payload_bytes: number;
  received_at: Date;
  metrics: Record<string, number>;
};

export type EventRow = {
  event_id: number;
  run_id: string | null;
  round_id: number | null;
  client_id: string | null;
  event_type: string;
  message_type: string | null;
  status: string | null;
  reason_code: string | null;
  occurred_at: Date;
  duration_ms: number | null;
  message_bytes: number | null;
  details: Record<string, unknown>;
};

const EVENT_COLUMNS = `event_id::float8 AS event_id, run_id, round_id, client_id, event_type,
  message_type, status, reason_code, occurred_at, duration_ms,
  message_bytes::float8 AS message_bytes, details`;

export async function listUpdates(runId: string, roundId: number) {
  if (!isUuid(runId)) return null;
  const accepted = await query<UpdateRow>(
    `SELECT update_id, client_id, model_version, num_samples, checksum_sha256,
       payload_bytes::float8 AS payload_bytes, received_at, metrics
     FROM updates WHERE run_id = $1 AND round_id = $2 ORDER BY received_at`,
    [runId, roundId],
  );
  const rejected = await query<EventRow>(
    `SELECT ${EVENT_COLUMNS} FROM events
     WHERE run_id = $1 AND round_id = $2 AND event_type IN ('UPDATE_REJECTED','UPDATE_EXCLUDED')
     ORDER BY occurred_at`,
    [runId, roundId],
  );
  return { run_id: runId, round_id: roundId, accepted, rejected };
}

export type EventFilter = {
  run_id?: string;
  round_id?: number;
  client_id?: string;
  event_type?: string;
  status?: string;
  limit: number;
  offset: number;
};

export async function listEvents(f: EventFilter) {
  const where: string[] = [];
  const params: unknown[] = [];
  const add = (column: string, value: unknown) => {
    params.push(value);
    where.push(`${column} = $${params.length}`);
  };
  if (f.run_id) add("run_id", f.run_id);
  if (f.round_id !== undefined) add("round_id", f.round_id);
  if (f.client_id) add("client_id", f.client_id);
  if (f.event_type) add("event_type", f.event_type);
  if (f.status) add("status", f.status);
  const clause = where.length ? `WHERE ${where.join(" AND ")}` : "";
  const [count] = await query<{ n: number }>(
    `SELECT count(*)::int AS n FROM events ${clause}`,
    params,
  );
  const events = await query<EventRow>(
    `SELECT ${EVENT_COLUMNS} FROM events ${clause}
     ORDER BY occurred_at DESC, event_id DESC LIMIT $${params.length + 1} OFFSET $${params.length + 2}`,
    [...params, f.limit, f.offset],
  );
  return { total: count.n, events };
}

export async function eventTypes(): Promise<string[]> {
  const rows = await query<{ event_type: string }>(
    "SELECT DISTINCT event_type FROM events ORDER BY event_type",
  );
  return rows.map((r) => r.event_type);
}

/** Coordinator metrics (no client) of one run, by round: the data behind the metric panels. */
export async function roundMetricSeries(runId: string, names: string[]) {
  if (!isUuid(runId)) return [];
  return query<{ round_id: number; metric_name: string; value: number }>(
    `SELECT round_id, metric_name, value FROM metrics
     WHERE run_id = $1 AND client_id IS NULL AND message_type IS NULL
       AND round_id IS NOT NULL AND metric_name = ANY($2) ORDER BY round_id`,
    [runId, names],
  );
}
