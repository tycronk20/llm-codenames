import { opSysPrompt } from '../prompts/opSysPrompt';
import { spySysPrompt } from '../prompts/spySysPrompt';
import { createUserPrompt } from '../prompts/userPrompt';
import { GameState, OperativeMove, Role, SpymasterMove } from './game';
import { modelCatalogById } from './modelCatalog';

export type OpenRouterReasoningDetail = Record<string, unknown>;

export type Message = {
  role: 'system' | 'user' | 'assistant';
  content: string;
  reasoningDetails?: OpenRouterReasoningDetail[];
};

export type AssistantPrefill = {
  content: string;
  reasoningDetails?: OpenRouterReasoningDetail[];
  reasoning?: string;
};

type LLMRequest = {
  messages: Message[];
  modelId: string;
  role: Role;
};

type LLMRequestOptions = {
  onIdleStateChange?: (message: string | null) => void;
};

const NETWORK_RETRY_DELAYS_MS = [250, 800];
const TOKEN_IDLE_TIMEOUT_MS = 15_000;

export type StreamedLLMResponse =
  | {
      type: 'progress';
      tokenCount: number;
    }
  | {
      type: 'prefill';
      prefill: AssistantPrefill;
    }
  | {
      type: 'reasoning';
      reasoning: string;
    }
  | {
      type: 'complete';
      move: SpymasterMove | OperativeMove;
    };

export function createMessagesFromGameState(
  gameState: GameState,
  assistantPrefill?: AssistantPrefill,
): Message[] {
  const messages: Message[] = [
    {
      role: 'system',
      content: gameState.currentRole === 'spymaster' ? spySysPrompt : opSysPrompt,
    },
    {
      role: 'user',
      content: createUserPrompt(gameState),
    },
  ];

  if (
    assistantPrefill &&
    (assistantPrefill.content.trim() || assistantPrefill.reasoningDetails?.length)
  ) {
    messages.push({
      role: 'assistant',
      content: assistantPrefill.content,
      ...(assistantPrefill.reasoningDetails?.length ?
        {
          reasoningDetails: assistantPrefill.reasoningDetails,
        }
      : {}),
    });
  }

  return messages;
}

export async function fetchLLMResponse(
  request: LLMRequest,
  signal?: AbortSignal,
  options: LLMRequestOptions = {},
): Promise<SpymasterMove | OperativeMove> {
  const idleMonitor = createIdleMonitor(request.modelId, options.onIdleStateChange);

  try {
    idleMonitor.reset();

    const response = await fetchWithRetry('/api/llm', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      signal,
    });

    idleMonitor.reset();
    const data = await parseResponseBody(response);
    if (!response.ok) {
      throw new Error(typeof data?.error === 'string' ? data.error : 'LLM request failed');
    }

    return data;
  } finally {
    idleMonitor.dispose();
  }
}

export async function* streamLLMResponse(
  request: LLMRequest,
  signal?: AbortSignal,
  options: LLMRequestOptions = {},
): AsyncGenerator<StreamedLLMResponse, SpymasterMove | OperativeMove, void> {
  let emittedReasoning = false;
  const idleMonitor = createIdleMonitor(request.modelId, options.onIdleStateChange);

  try {
    idleMonitor.reset();

    const response = await fetchWithRetry('/api/llm/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      signal,
    });

    idleMonitor.reset();
    if (!response.ok) {
      const data = await parseResponseBody(response);
      throw new Error(typeof data?.error === 'string' ? data.error : 'LLM stream request failed');
    }

    const reader = response.body?.getReader();
    if (!reader) {
      return yield* fallbackStream(request, signal, options);
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let completedMove: SpymasterMove | OperativeMove | null = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      idleMonitor.reset();
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

        if (event.type === 'progress' && typeof event.tokenCount === 'number') {
          yield {
            type: 'progress',
            tokenCount: event.tokenCount,
          };
        } else if (
          event.type === 'prefill' &&
          event.prefill &&
          typeof event.prefill === 'object' &&
          typeof (event.prefill as Record<string, unknown>).content === 'string'
        ) {
          const prefill = event.prefill as Record<string, unknown>;
          yield {
            type: 'prefill',
            prefill: {
              content: prefill.content as string,
              reasoning:
                typeof prefill.reasoning === 'string' ? prefill.reasoning : undefined,
              reasoningDetails:
                Array.isArray(prefill.reasoningDetails) ?
                  (prefill.reasoningDetails as OpenRouterReasoningDetail[])
                : undefined,
            },
          };
        } else if (event.type === 'reasoning' && typeof event.reasoning === 'string') {
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
      return yield* fallbackStream(request, signal, options, !emittedReasoning);
    }

    return completedMove;
  } catch (error) {
    if (isAbortLikeError(error)) {
      throw error;
    }

    return yield* fallbackStream(request, signal, options, !emittedReasoning);
  } finally {
    idleMonitor.dispose();
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
  options: LLMRequestOptions = {},
  emitReasoning = true,
): AsyncGenerator<StreamedLLMResponse, SpymasterMove | OperativeMove, void> {
  const move = await fetchLLMResponse(request, signal, options);
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

function createIdleMonitor(
  modelId: string,
  onIdleStateChange?: (message: string | null) => void,
) {
  const modelLabel = getModelLabel(modelId);
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  let isIdle = false;

  const clearIdleState = () => {
    if (!isIdle) {
      return;
    }

    isIdle = false;
    onIdleStateChange?.(null);
  };

  return {
    reset() {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      clearIdleState();
      timeoutId = setTimeout(() => {
        isIdle = true;
        onIdleStateChange?.(
          `No output from ${modelLabel} for ${TOKEN_IDLE_TIMEOUT_MS / 1000} seconds. Still waiting for a response...`,
        );
      }, TOKEN_IDLE_TIMEOUT_MS);
    },
    dispose() {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      clearIdleState();
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
