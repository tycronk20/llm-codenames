import { loadEnv } from 'vite';
import { startTurnWorker } from './llmProxy.ts';

const env = {
  ...process.env,
  ...loadEnv(process.env.NODE_ENV ?? 'development', process.cwd(), ''),
};

const worker = startTurnWorker(
  Object.fromEntries(
    Object.entries(env).filter((entry): entry is [string, string] => typeof entry[1] === 'string'),
  ),
);

console.log(`[llm-worker] started ${worker.workerId}`);

const shutdown = async (signal: string) => {
  console.log(`[llm-worker] stopping ${worker.workerId} on ${signal}`);
  await worker.stop();
  process.exit(0);
};

process.on('SIGINT', () => {
  void shutdown('SIGINT');
});

process.on('SIGTERM', () => {
  void shutdown('SIGTERM');
});
