import { appendFile, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import pg from 'pg';

export type DurableTurnStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'abandoned';

export type DurableTurnRecord = {
  id: string;
  requestId: string;
  status: DurableTurnStatus;
  createdAt: string;
  updatedAt: string;
  request: unknown;
  error?: string;
  workerId?: string;
  leaseExpiresAt?: string;
  startedAt?: string;
  completedAt?: string;
};

export type DurableTurnSnapshot = {
  record: DurableTurnRecord;
  events: string[];
  lastEventSeq: number;
};

export type TurnStore = {
  createTurn(record: DurableTurnRecord): Promise<boolean>;
  saveRecord(record: DurableTurnRecord): Promise<void>;
  appendEvent(id: string, rawEventLine: string): Promise<number>;
  load(id: string): Promise<DurableTurnSnapshot | null>;
  loadAfter(id: string, afterEventSeq: number): Promise<DurableTurnSnapshot | null>;
  claimNextQueuedTurn(workerId: string, leaseMs: number): Promise<DurableTurnRecord | null>;
  renewLease(id: string, workerId: string, leaseMs: number): Promise<boolean>;
  abandonExpiredTurn(id: string, error: string): Promise<boolean>;
};

type DbRow = {
  id: string;
  request_id: string;
  status: DurableTurnStatus;
  created_at: Date | string;
  updated_at: Date | string;
  request: unknown;
  error: string | null;
  worker_id: string | null;
  lease_expires_at: Date | string | null;
  started_at: Date | string | null;
  completed_at: Date | string | null;
};

const POSTGRES_STATUSES = ['queued', 'running', 'completed', 'failed', 'abandoned'] as const;

export function createTurnStoreFromEnv(
  env: Record<string, string>,
  fileBaseDirUrl: URL,
): TurnStore {
  if (env.DATABASE_URL?.trim()) {
    return createPostgresTurnStore(env.DATABASE_URL.trim());
  }

  return createFileTurnStore(fileBaseDirUrl);
}

export function createFileTurnStore(baseDirUrl: URL): TurnStore {
  let dirReady: Promise<void> | null = null;

  function getRecordUrl(id: string) {
    return new URL(`${id}.json`, baseDirUrl);
  }

  function getEventsUrl(id: string) {
    return new URL(`${id}.ndjson`, baseDirUrl);
  }

  async function ensureDir() {
    if (!dirReady) {
      dirReady = mkdir(baseDirUrl, { recursive: true }).then(() => undefined);
    }

    await dirReady;
  }

  async function readSnapshot(id: string) {
    await ensureDir();

    try {
      const [recordRaw, eventsRaw] = await Promise.all([
        readFile(getRecordUrl(id), 'utf8'),
        readFile(getEventsUrl(id), 'utf8').catch(() => ''),
      ]);

      const record = JSON.parse(recordRaw) as DurableTurnRecord;
      const events =
        eventsRaw ? eventsRaw.split('\n').filter(Boolean).map((line) => `${line}\n`) : [];

      return {
        record,
        events,
      };
    } catch {
      return null;
    }
  }

  return {
    async createTurn(record) {
      if (await readSnapshot(record.id)) {
        return false;
      }

      await ensureDir();
      await writeFile(getRecordUrl(record.id), JSON.stringify(record, null, 2));
      return true;
    },

    async saveRecord(record) {
      await ensureDir();
      await writeFile(getRecordUrl(record.id), JSON.stringify(record, null, 2));
    },

    async appendEvent(id, rawEventLine) {
      await ensureDir();
      await appendFile(getEventsUrl(id), rawEventLine);
      const snapshot = await readSnapshot(id);
      return snapshot?.events.length ?? 0;
    },

    async load(id) {
      const snapshot = await readSnapshot(id);
      if (!snapshot) {
        return null;
      }

      return {
        ...snapshot,
        lastEventSeq: snapshot.events.length,
      };
    },

    async loadAfter(id, afterEventSeq) {
      const snapshot = await readSnapshot(id);
      if (!snapshot) {
        return null;
      }

      return {
        record: snapshot.record,
        events: snapshot.events.slice(afterEventSeq),
        lastEventSeq: snapshot.events.length,
      };
    },

    async claimNextQueuedTurn(workerId, leaseMs) {
      await ensureDir();
      const entries = await readdir(baseDirUrl);
      const candidateIds = entries
        .filter((entry) => entry.endsWith('.json'))
        .map((entry) => entry.slice(0, -'.json'.length))
        .sort();

      for (const candidateId of candidateIds) {
        const snapshot = await readSnapshot(candidateId);
        if (!snapshot || snapshot.record.status !== 'queued') {
          continue;
        }

        snapshot.record.status = 'running';
        snapshot.record.workerId = workerId;
        snapshot.record.startedAt ??= new Date().toISOString();
        snapshot.record.updatedAt = new Date().toISOString();
        snapshot.record.leaseExpiresAt = new Date(Date.now() + leaseMs).toISOString();
        snapshot.record.error = undefined;
        await writeFile(
          getRecordUrl(candidateId),
          JSON.stringify(snapshot.record, null, 2),
        );
        return snapshot.record;
      }

      return null;
    },

    async renewLease(id, workerId, leaseMs) {
      const snapshot = await readSnapshot(id);
      if (!snapshot || snapshot.record.workerId !== workerId || snapshot.record.status !== 'running') {
        return false;
      }

      snapshot.record.updatedAt = new Date().toISOString();
      snapshot.record.leaseExpiresAt = new Date(Date.now() + leaseMs).toISOString();
      await writeFile(getRecordUrl(id), JSON.stringify(snapshot.record, null, 2));
      return true;
    },

    async abandonExpiredTurn(id, error) {
      const snapshot = await readSnapshot(id);
      if (!snapshot || snapshot.record.status !== 'running') {
        return false;
      }

      const leaseExpired =
        !snapshot.record.leaseExpiresAt ||
        new Date(snapshot.record.leaseExpiresAt).getTime() <= Date.now();
      if (!leaseExpired) {
        return false;
      }

      snapshot.record.status = 'abandoned';
      snapshot.record.error = error;
      snapshot.record.updatedAt = new Date().toISOString();
      snapshot.record.completedAt = snapshot.record.updatedAt;
      await writeFile(getRecordUrl(id), JSON.stringify(snapshot.record, null, 2));
      await appendFile(
        getEventsUrl(id),
        `${JSON.stringify({ type: 'error', error })}\n`,
      );
      return true;
    },
  };
}

export function createPostgresTurnStore(connectionString: string): TurnStore {
  const pool = new pg.Pool({
    connectionString,
    max: 10,
  });
  let ready: Promise<void> | null = null;

  async function ensureReady() {
    if (!ready) {
      ready = (async () => {
        await pool.query(`
          CREATE TABLE IF NOT EXISTS turns (
            id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status = ANY(ARRAY['queued','running','completed','failed','abandoned'])),
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            request JSONB NOT NULL,
            error TEXT,
            worker_id TEXT,
            lease_expires_at TIMESTAMPTZ,
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ
          )
        `);
        await pool.query(`
          CREATE TABLE IF NOT EXISTS turn_events (
            seq BIGSERIAL PRIMARY KEY,
            turn_id TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
            raw_event TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
          )
        `);
        await pool.query(`
          CREATE INDEX IF NOT EXISTS turns_status_created_idx
          ON turns (status, created_at)
        `);
        await pool.query(`
          CREATE INDEX IF NOT EXISTS turns_running_lease_idx
          ON turns (status, lease_expires_at)
        `);
        await pool.query(`
          CREATE INDEX IF NOT EXISTS turn_events_turn_seq_idx
          ON turn_events (turn_id, seq)
        `);
      })();
    }

    await ready;
  }

  return {
    async createTurn(record) {
      await ensureReady();
      const result = await pool.query(
        `
          INSERT INTO turns (
            id, request_id, status, created_at, updated_at, request, error, worker_id,
            lease_expires_at, started_at, completed_at
          )
          VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11)
          ON CONFLICT (id) DO NOTHING
        `,
        [
          record.id,
          record.requestId,
          record.status,
          record.createdAt,
          record.updatedAt,
          JSON.stringify(record.request),
          record.error ?? null,
          record.workerId ?? null,
          record.leaseExpiresAt ?? null,
          record.startedAt ?? null,
          record.completedAt ?? null,
        ],
      );
      return (result.rowCount ?? 0) > 0;
    },

    async saveRecord(record) {
      await ensureReady();
      await pool.query(
        `
          INSERT INTO turns (
            id, request_id, status, created_at, updated_at, request, error, worker_id,
            lease_expires_at, started_at, completed_at
          )
          VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11)
          ON CONFLICT (id) DO UPDATE SET
            request_id = EXCLUDED.request_id,
            status = EXCLUDED.status,
            updated_at = EXCLUDED.updated_at,
            request = EXCLUDED.request,
            error = EXCLUDED.error,
            worker_id = EXCLUDED.worker_id,
            lease_expires_at = EXCLUDED.lease_expires_at,
            started_at = EXCLUDED.started_at,
            completed_at = EXCLUDED.completed_at
        `,
        [
          record.id,
          record.requestId,
          record.status,
          record.createdAt,
          record.updatedAt,
          JSON.stringify(record.request),
          record.error ?? null,
          record.workerId ?? null,
          record.leaseExpiresAt ?? null,
          record.startedAt ?? null,
          record.completedAt ?? null,
        ],
      );
    },

    async appendEvent(id, rawEventLine) {
      await ensureReady();
      const result = await pool.query<{ seq: string }>(
        `
          INSERT INTO turn_events (turn_id, raw_event)
          VALUES ($1, $2)
          RETURNING seq
        `,
        [id, rawEventLine],
      );
      return Number(result.rows[0]?.seq ?? 0);
    },

    async load(id) {
      await ensureReady();
      const [turnResult, eventsResult] = await Promise.all([
        pool.query<DbRow>('SELECT * FROM turns WHERE id = $1', [id]),
        pool.query<{ seq: string; raw_event: string }>(
          'SELECT seq, raw_event FROM turn_events WHERE turn_id = $1 ORDER BY seq ASC',
          [id],
        ),
      ]);

      const row = turnResult.rows[0];
      if (!row) {
        return null;
      }

      return {
        record: mapTurnRow(row),
        events: eventsResult.rows.map((event) => event.raw_event),
        lastEventSeq: Number(eventsResult.rows.at(-1)?.seq ?? 0),
      };
    },

    async loadAfter(id, afterEventSeq) {
      await ensureReady();
      const [turnResult, eventsResult] = await Promise.all([
        pool.query<DbRow>('SELECT * FROM turns WHERE id = $1', [id]),
        pool.query<{ seq: string; raw_event: string }>(
          `
            SELECT seq, raw_event
            FROM turn_events
            WHERE turn_id = $1 AND seq > $2
            ORDER BY seq ASC
          `,
          [id, afterEventSeq],
        ),
      ]);

      const row = turnResult.rows[0];
      if (!row) {
        return null;
      }

      return {
        record: mapTurnRow(row),
        events: eventsResult.rows.map((event) => event.raw_event),
        lastEventSeq:
          eventsResult.rows.length > 0 ?
            Number(eventsResult.rows[eventsResult.rows.length - 1].seq)
          : afterEventSeq,
      };
    },

    async claimNextQueuedTurn(workerId, leaseMs) {
      await ensureReady();
      const client = await pool.connect();
      try {
        await client.query('BEGIN');
        const result = await client.query<DbRow>(
          `
            WITH candidate AS (
              SELECT id
              FROM turns
              WHERE status = 'queued'
              ORDER BY created_at ASC
              LIMIT 1
              FOR UPDATE SKIP LOCKED
            )
            UPDATE turns AS target
            SET
              status = 'running',
              worker_id = $1,
              updated_at = NOW(),
              started_at = COALESCE(target.started_at, NOW()),
              lease_expires_at = NOW() + make_interval(secs => $2::double precision / 1000.0),
              error = NULL
            FROM candidate
            WHERE target.id = candidate.id
            RETURNING target.*
          `,
          [workerId, leaseMs],
        );
        await client.query('COMMIT');
        return result.rows[0] ? mapTurnRow(result.rows[0]) : null;
      } catch (error) {
        await client.query('ROLLBACK');
        throw error;
      } finally {
        client.release();
      }
    },

    async renewLease(id, workerId, leaseMs) {
      await ensureReady();
      const result = await pool.query(
        `
          UPDATE turns
          SET
            updated_at = NOW(),
            lease_expires_at = NOW() + make_interval(secs => $3::double precision / 1000.0)
          WHERE id = $1 AND worker_id = $2 AND status = 'running'
        `,
        [id, workerId, leaseMs],
      );
      return (result.rowCount ?? 0) > 0;
    },

    async abandonExpiredTurn(id, error) {
      await ensureReady();
      const result = await pool.query<DbRow>(
        `
          UPDATE turns
          SET
            status = 'abandoned',
            error = $2,
            updated_at = NOW(),
            completed_at = NOW()
          WHERE
            id = $1
            AND status = 'running'
            AND lease_expires_at IS NOT NULL
            AND lease_expires_at <= NOW()
          RETURNING *
        `,
        [id, error],
      );

      if (!result.rowCount) {
        return false;
      }

      await pool.query(
        'INSERT INTO turn_events (turn_id, raw_event) VALUES ($1, $2)',
        [id, `${JSON.stringify({ type: 'error', error })}\n`],
      );
      return true;
    },
  };
}

function mapTurnRow(row: DbRow): DurableTurnRecord {
  const request =
    typeof row.request === 'string' ? JSON.parse(row.request) : row.request;

  return {
    id: row.id,
    requestId: row.request_id,
    status: normalizeStatus(row.status),
    createdAt: new Date(row.created_at).toISOString(),
    updatedAt: new Date(row.updated_at).toISOString(),
    request,
    ...(row.error ? { error: row.error } : {}),
    ...(row.worker_id ? { workerId: row.worker_id } : {}),
    ...(row.lease_expires_at ?
      { leaseExpiresAt: new Date(row.lease_expires_at).toISOString() }
    : {}),
    ...(row.started_at ? { startedAt: new Date(row.started_at).toISOString() } : {}),
    ...(row.completed_at ?
      { completedAt: new Date(row.completed_at).toISOString() }
    : {}),
  };
}

function normalizeStatus(status: string): DurableTurnStatus {
  return POSTGRES_STATUSES.includes(status as DurableTurnStatus) ?
      (status as DurableTurnStatus)
    : 'failed';
}
