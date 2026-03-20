import { execSync, spawn, type ChildProcess } from 'child_process';
import react from '@vitejs/plugin-react-swc';
import type { Plugin, PreviewServer, ViteDevServer } from 'vite';
import { defineConfig, loadEnv } from 'vite';

const DEFAULT_PORT = 5173;

function isPortInUse(port: number): boolean {
  try {
    execSync(`lsof -i :${port} -sTCP:LISTEN`, { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

function findFreePort(startPort: number): number {
  let port = startPort;
  while (isPortInUse(port)) {
    port++;
  }
  return port;
}

function portGuardPlugin(defaultPort: number): Plugin {
  return {
    name: 'port-guard',
    config() {
      const port = findFreePort(defaultPort);
      if (port !== defaultPort) {
        console.warn(
          `\n  Port ${defaultPort} is in use (another worktree?). Using port ${port} instead.\n`,
        );
        return { server: { port, strictPort: true }, preview: { port, strictPort: true } };
      }
    },
  };
}

function turnWorkerPlugin(): Plugin {
  let workerProcess: ChildProcess | null = null;

  const startWorker = () => {
    if (process.env.LLM_AUTO_START_WORKER === '0' || workerProcess) {
      return;
    }

    workerProcess = spawn('bun', ['server/turnWorker.ts'], {
      stdio: 'inherit',
      env: process.env,
    });

    const clearWorker = () => {
      workerProcess = null;
    };

    workerProcess.once('exit', clearWorker);
    workerProcess.once('close', clearWorker);
  };

  const stopWorker = () => {
    if (!workerProcess) {
      return;
    }

    workerProcess.kill('SIGTERM');
    workerProcess = null;
  };

  return {
    name: 'turn-worker',
    configureServer(server: ViteDevServer) {
      startWorker();
      server.httpServer?.once('close', stopWorker);
    },
    configurePreviewServer(server: PreviewServer) {
      startWorker();
      server.httpServer?.once('close', stopWorker);
    },
  };
}

export default defineConfig(async ({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const { createLlmApiMiddleware } = await import('./server/llmProxy.ts');
  const llmApiMiddleware = createLlmApiMiddleware(env);

  return {
    server: {
      port: DEFAULT_PORT,
      strictPort: true,
    },
    preview: {
      port: DEFAULT_PORT,
      strictPort: true,
    },
    plugins: [
      portGuardPlugin(DEFAULT_PORT),
      turnWorkerPlugin(),
      react(),
      {
        name: 'llm-api',
        configureServer(server: ViteDevServer) {
          server.middlewares.use(llmApiMiddleware);
        },
        configurePreviewServer(server: PreviewServer) {
          server.middlewares.use(llmApiMiddleware);
        },
      },
    ],
  };
});
