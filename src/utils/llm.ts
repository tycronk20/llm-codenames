import { opSysPrompt } from '../prompts/opSysPrompt';
import { spySysPrompt } from '../prompts/spySysPrompt';
import { createUserPrompt } from '../prompts/userPrompt';
import { GameState, OperativeMove, Role, SpymasterMove } from './game';
import { modelCatalogById } from './modelCatalog';

type Message = {
  role: 'system' | 'user' | 'assistant';
  content: string;
};

type LLMRequest = {
  messages: Message[];
  modelId: string;
  role: Role;
};

const NETWORK_RETRY_DELAYS_MS = [250, 800];
const TOKEN_IDLE_TIMEOUT_MS = 10_000;

export class LLMTokenTimeoutError extends Error {
  code = 'LLM_TOKEN_TIMEOUT';

  constructor(modelLabel: string, timeoutMs: number) {
    super(`${modelLabel} produced no tokens for ${timeoutMs / 1000} seconds. Turn aborted.`);
    this.name = 'LLMTokenTimeoutError';
  }
}

export type StreamedLLMResponse =
  | {
      type: 'reasoning';
      reasoning: string;
    }
  | {
      type: 'complete';
      move: SpymasterMove | OperativeMove;
    };

export function createMessagesFromGameState(gameState: GameState): Message[] {
  return [
    {
      role: 'system',
      content: gameState.currentRole === 'spymaster' ? spySysPrompt : opSysPrompt,
    },
    {
      role: 'user',
      content: createUserPrompt(gameState),
    },
  ];
}

export async function fetchLLMResponse(
  request: LLMRequest,
  signal?: AbortSignal,
): Promise<SpymasterMove | OperativeMove> {
  const timeout = createIdleTimeout(request.modelId, signal);

  try {
    timeout.reset();

    const response = await fetchWithRetry('/api/llm', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      signal: timeout.signal,
    });

    timeout.reset();
    const data = await parseResponseBody(response);
    if (!response.ok) {
      throw new Error(typeof data?.error === 'string' ? data.error : 'LLM request failed');
    }

    return data;
  } catch (error) {
    throw timeout.normalizeError(error);
  } finally {
    timeout.dispose();
  }
}

export async function* streamLLMResponse(
  request: LLMRequest,
  signal?: AbortSignal,
): AsyncGenerator<StreamedLLMResponse, SpymasterMove | OperativeMove, void> {
  let emittedReasoning = false;
  const timeout = createIdleTimeout(request.modelId, signal);

  try {
    timeout.reset();

    const response = await fetchWithRetry('/api/llm/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      signal: timeout.signal,
    });

    timeout.reset();
    if (!response.ok) {
      const data = await parseResponseBody(response);
      throw new Error(typeof data?.error === 'string' ? data.error : 'LLM stream request failed');
    }

    const reader = response.body?.getReader();
    if (!reader) {
      return yield* fallbackStream(request, signal);
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let completedMove: SpymasterMove | OperativeMove | null = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      timeout.reset();
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
          continue;
        }

        const event = safeJsonParse(trimmed);
        if (!event || typeof event !== 'object' || !('type' in event)) {
          continue;
        }

        if (event.type === 'reasoning' && typeof event.reasoning === 'string') {
          emittedReasoning = true;
          yield {
            type: 'reasoning',
            reasoning: event.reasoning,
          };
        } else if (event.type === 'complete' && event.move && typeof event.move === 'object') {
          completedMove = event.move as SpymasterMove | OperativeMove;
          yield {
            type: 'complete',
            move: completedMove,
          };
        } else if (event.type === 'error' && typeof event.error === 'string') {
          throw new Error(event.error);
        }
      }
    }

    if (!completedMove) {
      return yield* fallbackStream(request, signal, !emittedReasoning);
    }

    return completedMove;
  } catch (error) {
    const normalizedError = timeout.normalizeError(error);

    if (normalizedError instanceof LLMTokenTimeoutError || isAbortLikeError(normalizedError)) {
      throw normalizedError;
    }

    return yield* fallbackStream(request, signal, !emittedReasoning);
  } finally {
    timeout.dispose();
  }
}

async function parseResponseBody(response: Response) {
  const text = await response.text();
  if (!text) {
    return {};
  }

  try {
    return JSON.parse(text);
  } catch {
    return { error: text };
  }
}

async function* fallbackStream(
  request: LLMRequest,
  signal?: AbortSignal,
  emitReasoning = true,
): AsyncGenerator<StreamedLLMResponse, SpymasterMove | OperativeMove, void> {
  const move = await fetchLLMResponse(request, signal);
  if (emitReasoning) {
    yield {
      type: 'reasoning',
      reasoning: move.reasoning,
    };
  }
  yield {
    type: 'complete',
    move,
  };
  return move;
}

function safeJsonParse(text: string) {
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function createIdleTimeout(modelId: string, signal?: AbortSignal) {
  const controller = new AbortController();
  const modelLabel = getModelLabel(modelId);
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  let timeoutError: LLMTokenTimeoutError | null = null;

  const abortFromParent = () => {
    controller.abort(signal?.reason);
  };

  if (signal) {
    if (signal.aborted) {
      abortFromParent();
    } else {
      signal.addEventListener('abort', abortFromParent, { once: true });
    }
  }

  return {
    signal: controller.signal,
    reset() {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      timeoutId = setTimeout(() => {
        timeoutError = new LLMTokenTimeoutError(modelLabel, TOKEN_IDLE_TIMEOUT_MS);
        controller.abort(timeoutError);
      }, TOKEN_IDLE_TIMEOUT_MS);
    },
    normalizeError(error: unknown) {
      if (timeoutError && isAbortLikeError(error)) {
        return timeoutError;
      }

      return error;
    },
    dispose() {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      signal?.removeEventListener('abort', abortFromParent);
    },
  };
}

function getModelLabel(modelId: string) {
  const model = modelCatalogById[modelId];
  return model?.shortName || model?.modelName || `Model ${modelId}`;
}

function isAbortLikeError(error: unknown) {
  return (
    (error instanceof DOMException && error.name === 'AbortError') ||
    (error instanceof Error && error.name === 'AbortError')
  );
}

function isTransportError(error: unknown) {
  return (
    error instanceof TypeError ||
    (error instanceof Error && error.message === 'Failed to fetch')
  );
}

async function fetchWithRetry(input: RequestInfo | URL, init: RequestInit) {
  let lastError: unknown;

  for (let attempt = 0; attempt <= NETWORK_RETRY_DELAYS_MS.length; attempt++) {
    if (attempt > 0) {
      await sleep(NETWORK_RETRY_DELAYS_MS[attempt - 1]);
    }

    try {
      return await fetch(input, init);
    } catch (error) {
      if (isAbortLikeError(error) || !isTransportError(error) || attempt === NETWORK_RETRY_DELAYS_MS.length) {
        throw error;
      }

      lastError = error;
    }
  }

  throw lastError instanceof Error ? lastError : new Error('Failed to fetch');
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
