import react from '@vitejs/plugin-react-swc';
import type { PreviewServer, ViteDevServer } from 'vite';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(async ({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const { createLlmApiMiddleware } = await import('./server/llmProxy.ts');
  const llmApiMiddleware = createLlmApiMiddleware(env);

  return {
    plugins: [
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
