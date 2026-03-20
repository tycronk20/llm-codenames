import { randomUUID } from 'node:crypto';
import { appendFile, mkdir, writeFile } from 'node:fs/promises';
import { IncomingMessage, ServerResponse } from 'node:http';
import { createXai } from '@ai-sdk/xai';
import { generateText, streamText } from 'ai';
import { OpenRouter } from '@openrouter/sdk';
import { createRequire } from 'node:module';
import { jsonrepair } from 'jsonrepair';
import { modelCatalogById, type Provider, getOpenRouterReasoningEffort } from '../src/utils/modelCatalog.ts';
import { parsePromptContext, type GamePromptContext } from '../src/prompts/userPrompt.ts';

type Message = {
  role: 'system' | 'user' | 'assistant';
  content: string;
  reasoningDetails?: OpenRouterReasoningDetail[];
  reasoning?: string;
};

type MoveRole = 'spymaster' | 'operative';

type LlmApiRequest = {
  modelId: string;
  messages: Message[];
  role: MoveRole;
  promptContext?: GamePromptContext;
};

type JsonObject = Record<string, unknown>;
type OpenRouterReasoningDetail = Record<string, unknown>;
type AnthropicThinkingConfig = NonNullable<
  (typeof modelCatalogById)[string]['anthropicThinking']
>;
type AnthropicStreamEvent = {
  type?: string;
  index?: number;
  delta?: {
    type?: string;
    text?: string;
    thinking?: string;
    partial_json?: string;
    stop_reason?: string | null;
  };
  content_block?: {
    type?: string;
    text?: string;
    input?: unknown;
  };
  error?: {
    message?: string;
  };
};
type AnthropicTranscriptEvent = {
  seq: number;
  type: string;
  blockIndex?: number;
  blockType?: string;
  text?: string;
  json?: string;
  stopReason?: string;
};
type RequestLogEntry = {
  requestId: string;
  event: string;
  modelId?: string;
  provider?: string;
  role?: MoveRole;
  stream?: boolean;
  durationMs?: number;
  timeoutMs?: number;
  error?: string;
  messageCount?: number;
  messageChars?: number;
  systemChars?: number;
  userChars?: number;
  assistantChars?: number;
  responseKeys?: string[];
  reasoningChars?: number;
  guessCount?: number;
  clue?: string;
  number?: number;
  httpStatus?: number;
  sseEventCount?: number;
  outputChars?: number;
  stopReason?: string;
  guesses?: string[];
  gameSnapshot?: {
    currentTeam?: string;
    currentRole?: string;
    remainingRed?: number;
    remainingBlue?: number;
    clueText?: string;
    clueNumber?: number;
    cardCount?: number;
    revealedCount?: number;
    recentlyRevealedCount?: number;
    board?: Array<{
      word: string;
      color?: string;
      isRevealed?: boolean;
      wasRecentlyRevealed?: boolean;
    }>;
  };
};

const moveSchemas: Record<MoveRole, JsonObject> = {
  spymaster: {
    type: 'object',
    additionalProperties: false,
    required: ['reasoning', 'clue', 'number'],
    properties: {
      reasoning: { type: 'string' },
      clue: { type: 'string' },
      number: { type: 'integer', minimum: 1 },
    },
  },
  operative: {
    type: 'object',
    additionalProperties: false,
    required: ['reasoning', 'guesses'],
    properties: {
      reasoning: { type: 'string' },
      guesses: {
        type: 'array',
        items: { type: 'string' },
      },
    },
  },
};

const PROVIDER_TIMEOUT_MS: Record<'anthropic' | 'google' | 'openai' | 'openrouter' | 'xai', number> = {
  anthropic: 90_000,
  google: 75_000,
  openai: 90_000,
  openrouter: 90_000,
  xai: 90_000,
};
const REASONING_TIMEOUT_MS: Record<'medium' | 'high' | 'xhigh', number> = {
  medium: 90_000,
  high: 120_000,
  xhigh: 180_000,
};
const OPENROUTER_REASONING_TIMEOUT_MS = 150_000;
const LOG_DIR_URL = new URL('../logs/', import.meta.url);
const RUN_LOG_URL = new URL('../logs/llm-runs.jsonl', import.meta.url);
const ANTHROPIC_TRANSCRIPT_DIR_URL = new URL('../logs/anthropic-streams/', import.meta.url);

let logDirectoryReady: Promise<void> | null = null;
let anthropicTranscriptDirectoryReady: Promise<void> | null = null;
let processLoggingInstalled = false;
const require = createRequire(import.meta.url);
const GOOGLE_GENAI_MODULE = ['@google', 'genai'].join('/');
type StreamProviderHandler = (
  request: LlmApiRequest,
  env: Record<string, string>,
  res: ServerResponse<IncomingMessage>,
  requestId?: string,
) => Promise<void>;

const streamProviderHandlers: Partial<Record<Provider, StreamProviderHandler>> = {
  anthropic: streamAnthropicModel,
  google: streamGoogleModel,
  openai: streamOpenAiModel,
  openrouter: streamOpenRouterModel,
  xai: streamXaiModel,
};

export function createLlmApiMiddleware(env: Record<string, string>) {
  installProcessErrorLogging();

  return async (
    req: IncomingMessage,
    res: ServerResponse<IncomingMessage>,
    next: (err?: unknown) => void,
  ) => {
    if (req.method === 'POST' && req.url === '/api/llm/stream') {
      const requestId = randomUUID().slice(0, 8);
      const startedAt = Date.now();
      let request: LlmApiRequest | null = null;

      try {
        request = parseRequest(await readJsonBody(req));
        const model = modelCatalogById[request.modelId];
        const timeoutMs = getProviderTimeoutMs(request.modelId);

        await writeRunLog({
          requestId,
          event: 'request_started',
          modelId: request.modelId,
          provider: model?.provider,
          role: request.role,
          stream: true,
          timeoutMs,
          ...summarizeRequest(request),
        });

        await streamStructuredMove(request, env, res, requestId);

        await writeRunLog({
          requestId,
          event: 'request_succeeded',
          modelId: request.modelId,
          provider: model?.provider,
          role: request.role,
          stream: true,
          durationMs: Date.now() - startedAt,
        });
      } catch (error) {
        console.error('[llm-proxy] stream request failed:', error);
        await writeRunLog({
          requestId,
          event: 'request_failed',
          modelId: request?.modelId,
          provider: request ? modelCatalogById[request.modelId]?.provider : undefined,
          role: request?.role,
          stream: true,
          durationMs: Date.now() - startedAt,
          timeoutMs: request ? getProviderTimeoutMs(request.modelId) : undefined,
          error: getClientErrorMessage(
            error,
            request?.modelId,
            request ? getProviderTimeoutMs(request.modelId) : undefined,
          ),
          ...(request ? summarizeRequest(request) : {}),
        });
        if (!res.headersSent) {
          writeJson(res, isTimeoutError(error) ? 504 : 500, {
            error: getClientErrorMessage(error, request?.modelId, request ? getProviderTimeoutMs(request.modelId) : undefined),
          });
        } else {
          writeNdjsonEvent(res, {
            type: 'error',
            error: getClientErrorMessage(
              error,
              request?.modelId,
              request ? getProviderTimeoutMs(request.modelId) : undefined,
            ),
          });
          res.end();
        }
      }

      return;
    }

    if (req.method === 'POST' && req.url === '/api/config/active-models') {
      try {
        const body = await readJsonBody(req);
        if (!body || !Array.isArray(body.activeModelIds)) {
          throw new Error('Invalid request payload.');
        }

        const fs = await import('node:fs/promises');
        const catalogPath = new URL('../src/utils/modelCatalog.ts', import.meta.url);
        const content = await fs.readFile(catalogPath, 'utf8');

        const formattedIds = body.activeModelIds.map((id: string) => `  '${id}',`).join('\n');
        const newContent = content.replace(
          /export const activeModelIds: readonly ModelId\[\] = \[[^\]]*\];/,
          `export const activeModelIds: readonly ModelId[] = [\n${formattedIds}\n];`
        );

        await fs.writeFile(catalogPath, newContent, 'utf8');

        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ success: true }));
      } catch (error) {
        console.error('[llm-proxy] active models config failed:', error);
        res.statusCode = 500;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ error: String(error) }));
      }

      return;
    }

    if (req.method !== 'POST' || req.url !== '/api/llm') {
      next();
      return;
    }

    const requestId = randomUUID().slice(0, 8);
    const startedAt = Date.now();
    let request: LlmApiRequest | null = null;

    try {
      request = parseRequest(await readJsonBody(req));
      const model = modelCatalogById[request.modelId];
      const timeoutMs = getProviderTimeoutMs(request.modelId);

      await writeRunLog({
        requestId,
        event: 'request_started',
        modelId: request.modelId,
        provider: model?.provider,
        role: request.role,
        stream: false,
        timeoutMs,
        ...summarizeRequest(request),
      });

      const result = await fetchStructuredMove(request, env, requestId);
      const durationMs = Date.now() - startedAt;

      await writeRunLog({
        requestId,
        event: 'request_succeeded',
        modelId: request.modelId,
        provider: model?.provider,
        role: request.role,
        stream: false,
        durationMs,
        timeoutMs,
        ...summarizeMove(result),
      });

      writeJson(res, 200, result);
    } catch (error) {
      const model = request ? modelCatalogById[request.modelId] : undefined;
      const timeoutMs = request ? getProviderTimeoutMs(request.modelId) : undefined;
      const durationMs = Date.now() - startedAt;
      const errorMessage = getClientErrorMessage(error, request?.modelId, timeoutMs);
      const statusCode = isTimeoutError(error) ? 504 : 500;

      console.error(
        `[llm-proxy] request ${requestId} failed after ${durationMs}ms:`,
        error instanceof Error ? error.message : error,
      );
      await writeRunLog({
        requestId,
        event: 'request_failed',
        modelId: request?.modelId,
        provider: model?.provider,
        role: request?.role,
        stream: false,
        durationMs,
        timeoutMs,
        error: errorMessage,
        ...(request ? summarizeRequest(request) : {}),
      });
      writeJson(res, statusCode, {
        error: errorMessage,
        requestId,
      });
    }
  };
}

function installProcessErrorLogging() {
  if (processLoggingInstalled) {
    return;
  }

  processLoggingInstalled = true;

  process.on('uncaughtException', (error) => {
    void writeRunLog({
      requestId: 'process',
      event: 'uncaught_exception',
      error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    });
  });

  process.on('unhandledRejection', (reason) => {
    void writeRunLog({
      requestId: 'process',
      event: 'unhandled_rejection',
      error: reason instanceof Error ? `${reason.name}: ${reason.message}` : String(reason),
    });
  });
}

async function fetchStructuredMove(
  request: LlmApiRequest,
  env: Record<string, string>,
  requestId?: string,
) {
  const model = modelCatalogById[request.modelId];
  if (!model) {
    throw new Error(`Unknown model id: ${request.modelId}`);
  }

  const timeoutMs = getProviderTimeoutMs(request.modelId);
  const startedAt = Date.now();

  if (requestId) {
    await writeRunLog({
      requestId,
      event: 'provider_started',
      modelId: request.modelId,
      provider: model.provider,
      role: request.role,
      timeoutMs,
    });
  }

  try {
    const rawResult =
      model.provider === 'openai' ? await callOpenAiModel(request, env)
        : model.provider === 'anthropic' ? await callAnthropicModel(request, env)
          : model.provider === 'google' ? await callGoogleModel(request, env)
            : model.provider === 'xai' ? await callXaiModel(request, env, requestId)
              : await callOpenRouterModel(request, env, requestId);

    if (requestId) {
      await writeRunLog({
        requestId,
        event: 'provider_succeeded',
        modelId: request.modelId,
        provider: model.provider,
        role: request.role,
        durationMs: Date.now() - startedAt,
        timeoutMs,
        ...summarizeRawResult(rawResult),
      });
    }

    const move = normalizeMove(request.role, rawResult);

    if (requestId) {
      await writeRunLog({
        requestId,
        event: 'move_normalized',
        modelId: request.modelId,
        provider: model.provider,
        role: request.role,
        ...summarizeMove(move),
      });
    }

    return move;
  } catch (error) {
    if (requestId) {
      await writeRunLog({
        requestId,
        event: 'provider_failed',
        modelId: request.modelId,
        provider: model.provider,
        role: request.role,
        durationMs: Date.now() - startedAt,
        timeoutMs,
        error: getClientErrorMessage(error, request.modelId, timeoutMs),
      });
    }

    throw error;
  }
}

async function streamStructuredMove(
  request: LlmApiRequest,
  env: Record<string, string>,
  res: ServerResponse<IncomingMessage>,
  requestId?: string,
) {
  const model = modelCatalogById[request.modelId];
  if (!model) {
    throw new Error(`Unknown model id: ${request.modelId}`);
  }

  const streamHandler = streamProviderHandlers[model.provider];
  if (streamHandler) {
    try {
      await streamHandler(request, env, res, requestId);
    } catch (error) {
      if (res.writableEnded) {
        throw error;
      }

      if (requestId) {
        await writeRunLog({
          requestId,
          event: 'stream_handler_recovering',
          modelId: request.modelId,
          provider: model.provider,
          role: request.role,
          stream: true,
          error: getClientErrorMessage(error, request.modelId, getProviderTimeoutMs(request.modelId)),
        });
      }

      await streamResolvedMove(request, env, res, requestId, error);
    }
    return;
  }

  await streamResolvedMove(request, env, res, requestId);
}

async function streamResolvedMove(
  request: LlmApiRequest,
  env: Record<string, string>,
  res: ServerResponse<IncomingMessage>,
  requestId?: string,
  originalError?: unknown,
) {
  const move = await fetchStructuredMove(request, env, requestId);
  const shouldEmitReasoning = !res.headersSent;

  if (!res.headersSent) {
    openNdjsonStream(res);
    writeStreamOpenEvent(res);
  }

  if (requestId && originalError) {
    await writeRunLog({
      requestId,
      event: 'stream_recovered_with_nonstream_move',
      modelId: request.modelId,
      provider: modelCatalogById[request.modelId]?.provider,
      role: request.role,
      stream: true,
      error: getClientErrorMessage(
        originalError,
        request.modelId,
        getProviderTimeoutMs(request.modelId),
      ),
      ...summarizeMove(move),
    });
  }

  if (shouldEmitReasoning) {
    writeNdjsonEvent(res, {
      type: 'reasoning',
      reasoning: move.reasoning,
    });
  }

  writeNdjsonEvent(res, {
    type: 'complete',
    move,
  });
  res.end();
}

async function callOpenAiModel(request: LlmApiRequest, env: Record<string, string>) {
  const client = createOpenAiClient(request.modelId, env);
  const data = await client.responses.create(createOpenAiResponseBody(request), {
    timeout: getProviderTimeoutMs(request.modelId),
  });
  return parseOpenAiResponsesPayload(data);
}

async function callAnthropicModel(request: LlmApiRequest, env: Record<string, string>) {
  const timeoutMs = getProviderTimeoutMs(request.modelId);
  const response = await fetchWithTimeout(
    'https://api.anthropic.com/v1/messages',
    createAnthropicRequestInit(request, env),
    timeoutMs,
  );

  const data = await readApiResponse(response, 'Anthropic');
  const toolUse = Array.isArray(data.content) ?
    data.content.find((item: { type?: string }) => item.type === 'tool_use')
    : undefined;

  if (toolUse?.input) {
    return toolUse.input;
  }

  const textBlock = Array.isArray(data.content) ?
    data.content.find((item: { type?: string }) => item.type === 'text')
    : undefined;

  return parseJsonContent(textBlock?.text);
}

async function callGoogleModel(request: LlmApiRequest, env: Record<string, string>) {
  const client = createGoogleGenAiClient(request.modelId, env);
  const response = await client.models.generateContent(
    createGoogleGenerateContentParams(request, { includeThoughts: true }),
  );

  return parseGoogleGenerateContentPayload(response);
}

async function callXaiModel(
  request: LlmApiRequest,
  env: Record<string, string>,
  requestId?: string,
) {
  const response = await generateText({
    ...createXaiTextParams(request, env),
    abortSignal: AbortSignal.timeout(getProviderTimeoutMs(request.modelId)),
  });

  return resolveMoveWithFormatRepair(request, env, {
    requestId,
    source: 'xai_nonstream',
    output: response.text,
    reasoning: response.reasoningText?.trim(),
  });
}

async function callOpenRouterModel(
  request: LlmApiRequest,
  env: Record<string, string>,
  requestId?: string,
) {
  const client = createOpenRouterClient(request.modelId, env);
  const response = await client.chat.send(
    {
      chatGenerationParams:
        createOpenRouterChatParams(request) as import('@openrouter/sdk/models').ChatGenerationParams & {
          stream?: false;
        },
    },
    {
      timeoutMs: getProviderTimeoutMs(request.modelId),
    },
  );

  return resolveOpenRouterMove(request, env, response, requestId);
}

async function streamOpenAiModel(
  request: LlmApiRequest,
  env: Record<string, string>,
  res: ServerResponse<IncomingMessage>,
  requestId?: string,
) {
  const model = modelCatalogById[request.modelId];
  const timeoutMs = getProviderTimeoutMs(request.modelId);
  const startedAt = Date.now();
  if (requestId) {
    await writeRunLog({
      requestId,
      event: 'provider_started',
      modelId: request.modelId,
      provider: model?.provider,
      role: request.role,
      stream: true,
      timeoutMs,
    });
  }
  const client = createOpenAiClient(request.modelId, env);
  const stream = client.responses.stream(createOpenAiResponseBody(request), {
    timeout: timeoutMs,
  });
  openNdjsonStream(res);
  writeStreamOpenEvent(res);
  const lastMessage = request.messages[request.messages.length - 1];
  const isPrefill = lastMessage?.role === 'assistant';
  let reasoningText = isPrefill ? lastMessage.reasoning || '' : '';
  let streamedText = isPrefill ? `${lastMessage.reasoning || ''}${lastMessage.content || ''}` : '';
  let sseEventCount = 0;
  try {
    for await (const event of stream) {
      sseEventCount++;
      if (event.type === 'response.reasoning_summary_text.delta') {
        reasoningText += event.delta;
        streamedText += event.delta;
        writeProgressEvent(res, streamedText);
        writeNdjsonEvent(res, {
          type: 'reasoning',
          reasoning: reasoningText,
        });
      } else if (event.type === 'response.output_text.delta' && typeof event.delta === 'string') {
        streamedText += event.delta;
        writeProgressEvent(res, streamedText);
      }
    }

    const finalResponse = await stream.finalResponse();
    const parsedMove = parseOpenAiResponsesPayload(finalResponse);
    const move = normalizeMove(request.role, {
      ...parsedMove,
      reasoning: reasoningText || parsedMove.reasoning,
    });

    if (requestId) {
      await writeRunLog({
        requestId,
        event: 'provider_succeeded',
        modelId: request.modelId,
        provider: model?.provider,
        role: request.role,
        stream: true,
        durationMs: Date.now() - startedAt,
        timeoutMs,
        sseEventCount,
        ...summarizeMove(move),
      });
    }

    writeNdjsonEvent(res, {
      type: 'complete',
      move,
    });
    res.end();
  } catch (error) {
    if (requestId) {
      await writeRunLog({
        requestId,
        event: 'provider_failed',
        modelId: request.modelId,
        provider: model?.provider,
        role: request.role,
        stream: true,
        durationMs: Date.now() - startedAt,
        timeoutMs,
        sseEventCount,
        reasoningChars: reasoningText.length,
        error: getClientErrorMessage(error, request.modelId, timeoutMs),
      });
    }

    throw error;
  } finally {
    stream.abort();
  }
}

async function streamAnthropicModel(
  request: LlmApiRequest,
  env: Record<string, string>,
  res: ServerResponse<IncomingMessage>,
  requestId?: string,
) {
  const model = modelCatalogById[request.modelId];
  const timeoutMs = getProviderTimeoutMs(request.modelId);
  const startedAt = Date.now();
  if (requestId) {
    await writeRunLog({
      requestId,
      event: 'provider_started',
      modelId: request.modelId,
      provider: model?.provider,
      role: request.role,
      stream: true,
      timeoutMs,
    });
  }

  const response = await fetchWithTimeout(
    'https://api.anthropic.com/v1/messages',
    createAnthropicRequestInit(request, env, {
      stream: true,
      includeThinking: true,
    }),
    timeoutMs,
  );

  if (!response.ok) {
    throw await createApiError(response, 'Anthropic');
  }

  if (!response.body) {
    throw new Error('Anthropic stream response was empty.');
  }

  openNdjsonStream(res);
  writeStreamOpenEvent(res);
  const lastMessage = request.messages[request.messages.length - 1];
  const isPrefill = lastMessage?.role === 'assistant';
  let reasoningText = isPrefill ? lastMessage.reasoning || '' : '';
  let assistantText = isPrefill ? lastMessage.content || '' : '';
  let streamedText = isPrefill ? `${lastMessage.reasoning || ''}${lastMessage.content || ''}` : '';
  let streamedToolInput: Record<string, unknown> | null = null;
  let sseEventCount = 0;
  let stopReason = '';
  const transcriptEvents: AnthropicTranscriptEvent[] = [];
  const contentBlocks = new Map<number, { type?: string; text: string; partialJson: string }>();

  try {
    for await (const data of readSseData(response.body)) {
      sseEventCount++;

      const event = safeJsonParse(data) as AnthropicStreamEvent;
      if (!event || typeof event !== 'object') {
        continue;
      }

      const blockIndex = typeof event.index === 'number' ? event.index : undefined;

      if (event.type === 'error') {
        transcriptEvents.push({
          seq: sseEventCount,
          type: 'error',
          text: event.error?.message ?? 'Anthropic stream failed.',
        });
        throw new Error(event.error?.message ?? 'Anthropic stream failed.');
      }

      if (event.type === 'message_delta' && typeof event.delta?.stop_reason === 'string') {
        stopReason = event.delta.stop_reason;
        transcriptEvents.push({
          seq: sseEventCount,
          type: 'message_delta',
          stopReason,
        });
        continue;
      }

      if (event.type === 'content_block_start' && blockIndex !== undefined) {
        const initialInput =
          event.content_block?.input &&
            typeof event.content_block.input === 'object' &&
            Object.keys(event.content_block.input as Record<string, unknown>).length > 0 ?
            JSON.stringify(event.content_block.input)
            : '';
        contentBlocks.set(blockIndex, {
          type: event.content_block?.type,
          text: typeof event.content_block?.text === 'string' ? event.content_block.text : '',
          partialJson: event.content_block?.type === 'tool_use' ? initialInput : '',
        });
        transcriptEvents.push({
          seq: sseEventCount,
          type: 'content_block_start',
          blockIndex,
          blockType: event.content_block?.type,
          text: typeof event.content_block?.text === 'string' ? event.content_block.text : undefined,
          json: initialInput || undefined,
        });
        continue;
      }

      if (event.type === 'content_block_delta' && blockIndex !== undefined) {
        const block = contentBlocks.get(blockIndex);
        if (!block) {
          continue;
        }

        if (event.delta?.type === 'thinking_delta' && typeof event.delta.thinking === 'string') {
          reasoningText += event.delta.thinking;
          streamedText += event.delta.thinking;
          transcriptEvents.push({
            seq: sseEventCount,
            type: 'thinking_delta',
            blockIndex,
            blockType: block.type,
            text: event.delta.thinking,
          });
          writeProgressEvent(res, streamedText);
          writeNdjsonEvent(res, {
            type: 'reasoning',
            reasoning: reasoningText,
          });
          continue;
        }

        if (event.delta?.type === 'text_delta' && typeof event.delta.text === 'string') {
          block.text += event.delta.text;
          streamedText += event.delta.text;
          transcriptEvents.push({
            seq: sseEventCount,
            type: 'text_delta',
            blockIndex,
            blockType: block.type,
            text: event.delta.text,
          });
          writeProgressEvent(res, streamedText);
          continue;
        }

        if (
          event.delta?.type === 'input_json_delta' &&
          typeof event.delta.partial_json === 'string'
        ) {
          block.partialJson += event.delta.partial_json;
          streamedText += event.delta.partial_json;
          transcriptEvents.push({
            seq: sseEventCount,
            type: 'input_json_delta',
            blockIndex,
            blockType: block.type,
            json: event.delta.partial_json,
          });
          writeProgressEvent(res, streamedText);
        }
        continue;
      }

      if (event.type === 'content_block_stop' && blockIndex !== undefined) {
        const block = contentBlocks.get(blockIndex);
        if (!block) {
          continue;
        }

        if (block.type === 'tool_use' && block.partialJson.trim()) {
          streamedToolInput = parseJsonContent(block.partialJson) as Record<string, unknown>;
        } else if (block.type === 'text' && block.text.trim()) {
          assistantText += `${assistantText ? '\n' : ''}${block.text.trim()}`;
        }

        transcriptEvents.push({
          seq: sseEventCount,
          type: 'content_block_stop',
          blockIndex,
          blockType: block.type,
          text: block.text || undefined,
          json: block.partialJson || undefined,
        });

        contentBlocks.delete(blockIndex);
      }
    }

    const rawMove =
      streamedToolInput ??
      (assistantText.trim() ? (parseJsonContent(assistantText) as Record<string, unknown>) : null);
    const resolvedRawMove =
      rawMove ?? (await recoverAnthropicMoveFromNonStream(request, env, requestId, stopReason));

    const streamedReasoning = reasoningText.trim();
    const move = normalizeMove(request.role, {
      ...resolvedRawMove,
      reasoning: streamedReasoning || resolvedRawMove.reasoning,
    });

    if (requestId) {
      await writeRunLog({
        requestId,
        event: 'provider_succeeded',
        modelId: request.modelId,
        provider: model?.provider,
        role: request.role,
        stream: true,
        durationMs: Date.now() - startedAt,
        timeoutMs,
        sseEventCount,
        outputChars: assistantText.length,
        stopReason: stopReason || undefined,
        ...summarizeMove(move),
      });
    }

    if (!streamedReasoning) {
      writeNdjsonEvent(res, {
        type: 'reasoning',
        reasoning: move.reasoning,
      });
    }

    writeNdjsonEvent(res, {
      type: 'complete',
      move,
    });
    res.end();
    await writeAnthropicTranscript({
      requestId,
      modelId: request.modelId,
      role: request.role,
      startedAt: new Date(startedAt).toISOString(),
      finishedAt: new Date().toISOString(),
      status: 'succeeded',
      stopReason: stopReason || undefined,
      messageSummary: summarizeRequest(request),
      reasoningText,
      assistantText,
      streamedToolInput,
      move,
      events: transcriptEvents,
    });
  } catch (error) {
    if (requestId) {
      await writeRunLog({
        requestId,
        event: 'provider_failed',
        modelId: request.modelId,
        provider: model?.provider,
        role: request.role,
        stream: true,
        durationMs: Date.now() - startedAt,
        timeoutMs,
        sseEventCount,
        reasoningChars: reasoningText.length,
        outputChars: assistantText.length,
        stopReason: stopReason || undefined,
        error: getClientErrorMessage(error, request.modelId, timeoutMs),
      });
    }

    await writeAnthropicTranscript({
      requestId,
      modelId: request.modelId,
      role: request.role,
      startedAt: new Date(startedAt).toISOString(),
      finishedAt: new Date().toISOString(),
      status: 'failed',
      stopReason: stopReason || undefined,
      error: getClientErrorMessage(error, request.modelId, timeoutMs),
      messageSummary: summarizeRequest(request),
      reasoningText,
      assistantText,
      streamedToolInput,
      events: transcriptEvents,
    });

    throw error;
  }
}

async function recoverAnthropicMoveFromNonStream(
  request: LlmApiRequest,
  env: Record<string, string>,
  requestId?: string,
  stopReason?: string,
) {
  if (requestId) {
    await writeRunLog({
      requestId,
      event: 'provider_fallback_nonstream',
      modelId: request.modelId,
      provider: 'anthropic',
      role: request.role,
      stream: true,
      stopReason: stopReason || 'missing_move',
    });
  }

  return callAnthropicModel(request, env);
}

async function recoverOpenRouterMoveFromNonStream(
  request: LlmApiRequest,
  env: Record<string, string>,
  requestId?: string,
  reason?: string,
) {
  if (requestId) {
    await writeRunLog({
      requestId,
      event: 'provider_fallback_nonstream',
      modelId: request.modelId,
      provider: 'openrouter',
      role: request.role,
      stream: true,
      stopReason: reason || 'missing_move',
    });
  }

  const rawMove = await callOpenRouterModel(request, env, requestId);
  return normalizeMove(request.role, rawMove);
}

async function streamGoogleModel(
  request: LlmApiRequest,
  env: Record<string, string>,
  res: ServerResponse<IncomingMessage>,
  requestId?: string,
) {
  const model = modelCatalogById[request.modelId];
  const timeoutMs = getProviderTimeoutMs(request.modelId);
  const startedAt = Date.now();
  if (requestId) {
    await writeRunLog({
      requestId,
      event: 'provider_started',
      modelId: request.modelId,
      provider: model?.provider,
      role: request.role,
      stream: true,
      timeoutMs,
    });
  }

  const client = createGoogleGenAiClient(request.modelId, env);
  const stream = await client.models.generateContentStream(
    createGoogleGenerateContentParams(request, { includeThoughts: true }),
  );
  openNdjsonStream(res);
  writeStreamOpenEvent(res);
  const lastMessage = request.messages[request.messages.length - 1];
  const isPrefill = lastMessage?.role === 'assistant';
  let reasoningText = isPrefill ? lastMessage.reasoning || '' : '';
  let outputText = isPrefill ? lastMessage.content || '' : '';
  let streamedText = isPrefill ? `${lastMessage.reasoning || ''}${lastMessage.content || ''}` : '';
  let sseEventCount = 0;

  try {
    for await (const chunk of stream) {
      sseEventCount++;

      const reasoningDelta = extractGoogleThoughtText(chunk, '');
      if (reasoningDelta) {
        reasoningText += reasoningDelta;
        streamedText += reasoningDelta;
        writeProgressEvent(res, streamedText);
        writeNdjsonEvent(res, {
          type: 'reasoning',
          reasoning: reasoningText.trimEnd(),
        });
      }

      const outputDelta = getGoogleResponseText(chunk);
      if (outputDelta) {
        outputText += outputDelta;
        streamedText += outputDelta;
        writeProgressEvent(res, streamedText);
      }
    }

    const move = await resolveMoveWithFormatRepair(request, env, {
      requestId,
      source: 'google_stream',
      output: outputText,
      reasoning: reasoningText.trim(),
    });

    if (requestId) {
      await writeRunLog({
        requestId,
        event: 'provider_succeeded',
        modelId: request.modelId,
        provider: model?.provider,
        role: request.role,
        stream: true,
        durationMs: Date.now() - startedAt,
        timeoutMs,
        sseEventCount,
        outputChars: outputText.length,
        ...summarizeMove(move),
      });
    }

    if (!reasoningText.trim()) {
      writeNdjsonEvent(res, {
        type: 'reasoning',
        reasoning: move.reasoning,
      });
    }

    writeNdjsonEvent(res, {
      type: 'complete',
      move,
    });
    res.end();
  } catch (error) {
    if (requestId) {
      await writeRunLog({
        requestId,
        event: 'provider_failed',
        modelId: request.modelId,
        provider: model?.provider,
        role: request.role,
        stream: true,
        durationMs: Date.now() - startedAt,
        timeoutMs,
        sseEventCount,
        reasoningChars: reasoningText.length,
        outputChars: outputText.length,
        error: getClientErrorMessage(error, request.modelId, timeoutMs),
      });
    }

    throw error;
  }
}

function hasXaiEncryptedReasoningMetadata(part: { providerMetadata?: unknown }) {
  const meta = part.providerMetadata;
  if (!meta || typeof meta !== 'object') {
    return false;
  }

  const xai = (meta as Record<string, unknown>).xai;
  if (!xai || typeof xai !== 'object') {
    return false;
  }

  const encrypted = (xai as Record<string, unknown>).reasoningEncryptedContent;
  return typeof encrypted === 'string' ? encrypted.length > 0 : encrypted != null;
}

async function streamXaiModel(
  request: LlmApiRequest,
  env: Record<string, string>,
  res: ServerResponse<IncomingMessage>,
  requestId?: string,
) {
  const model = modelCatalogById[request.modelId];
  const timeoutMs = getProviderTimeoutMs(request.modelId);
  const startedAt = Date.now();
  if (requestId) {
    await writeRunLog({
      requestId,
      event: 'provider_started',
      modelId: request.modelId,
      provider: model?.provider,
      role: request.role,
      stream: true,
      timeoutMs,
    });
  }

  const lastMessage = request.messages[request.messages.length - 1];
  const isPrefill = lastMessage?.role === 'assistant';
  let reasoningText = isPrefill ? lastMessage.reasoning || '' : '';
  let outputText = isPrefill ? lastMessage.content || '' : '';
  let streamedText = isPrefill ? `${lastMessage.reasoning || ''}${lastMessage.content || ''}` : '';
  let sseEventCount = 0;

  const result = streamText({
    ...createXaiTextParams(request, env),
    abortSignal: AbortSignal.timeout(timeoutMs),
  });

  openNdjsonStream(res);
  writeStreamOpenEvent(res);

  try {
    for await (const part of result.fullStream) {
      sseEventCount++;

      if (part.type === 'reasoning-start') {
        writeNdjsonEvent(res, {
          type: 'reasoning_phase',
          active: true,
        });
        continue;
      }

      if (part.type === 'reasoning-end') {
        writeNdjsonEvent(res, {
          type: 'reasoning_phase',
          active: false,
          encrypted: hasXaiEncryptedReasoningMetadata(part),
        });
        continue;
      }

      if (part.type === 'finish-step') {
        const rt = part.usage?.outputTokenDetails?.reasoningTokens;
        if (typeof rt === 'number' && rt > 0) {
          writeNdjsonEvent(res, {
            type: 'reasoning_usage',
            reasoningTokens: rt,
          });
        }
        continue;
      }

      if (part.type === 'reasoning-delta') {
        const delta = part.text || '';
        if (!delta) {
          continue;
        }

        reasoningText += delta;
        streamedText += delta;
        writeProgressEvent(res, streamedText);
        writeNdjsonEvent(res, {
          type: 'reasoning',
          reasoning: reasoningText.trimEnd(),
        });
        continue;
      }

      if (part.type === 'text-delta') {
        const delta = part.text || '';
        if (!delta) {
          continue;
        }

        outputText += delta;
        streamedText += delta;
        writeProgressEvent(res, streamedText);
      }
    }

    const finalReasoning = (await result.reasoningText)?.trim() || reasoningText.trim();
    const finalText = (await result.text) || outputText;
    const move = await resolveMoveWithFormatRepair(request, env, {
      requestId,
      source: 'xai_stream',
      output: finalText,
      reasoning: finalReasoning,
    });

    if (requestId) {
      await writeRunLog({
        requestId,
        event: 'provider_succeeded',
        modelId: request.modelId,
        provider: model?.provider,
        role: request.role,
        stream: true,
        durationMs: Date.now() - startedAt,
        timeoutMs,
        sseEventCount,
        reasoningChars: finalReasoning.length,
        outputChars: finalText.length,
        ...summarizeMove(move),
      });
    }

    writeNdjsonEvent(res, {
      type: 'reasoning',
      reasoning: finalReasoning || move.reasoning,
    });
    writeNdjsonEvent(res, {
      type: 'complete',
      move,
    });
    res.end();
  } catch (error) {
    if (requestId) {
      await writeRunLog({
        requestId,
        event: 'provider_failed',
        modelId: request.modelId,
        provider: model?.provider,
        role: request.role,
        stream: true,
        durationMs: Date.now() - startedAt,
        timeoutMs,
        sseEventCount,
        reasoningChars: reasoningText.length,
        outputChars: outputText.length,
        error: getClientErrorMessage(error, request.modelId, timeoutMs),
      });
    }

    throw error;
  }
}

async function streamOpenRouterModel(
  request: LlmApiRequest,
  env: Record<string, string>,
  res: ServerResponse<IncomingMessage>,
  requestId?: string,
) {
  const model = modelCatalogById[request.modelId];
  const client = createOpenRouterClient(request.modelId, env);
  const timeoutMs = getProviderTimeoutMs(request.modelId);
  const startedAt = Date.now();
  if (requestId) {
    await writeRunLog({
      requestId,
      event: 'provider_started',
      modelId: request.modelId,
      provider: model?.provider,
      role: request.role,
      stream: true,
      timeoutMs,
    });
  }

  const inactivityTimeout = createInactivityTimeoutController(request.modelId, timeoutMs);
  const lastMessage = request.messages[request.messages.length - 1];
  const isPrefill = lastMessage?.role === 'assistant';
  let reasoningText = isPrefill ? lastMessage.reasoning || '' : '';
  let outputText = isPrefill ? lastMessage.content || '' : '';
  const reasoningDetails: OpenRouterReasoningDetail[] = isPrefill && lastMessage.reasoningDetails ? lastMessage.reasoningDetails : [];
  let streamedText = isPrefill ? `${lastMessage.reasoning || ''}${lastMessage.content || ''}` : '';
  let sseEventCount = 0;
  try {
    inactivityTimeout.reset();
    const streamOrResponse = await client.chat.send(
      {
        chatGenerationParams:
          createOpenRouterChatParams(request, { stream: true }) as import('@openrouter/sdk/models').ChatGenerationParams & {
            stream: true;
          },
      },
      {
        timeoutMs,
        signal: inactivityTimeout.signal,
      },
    );

    inactivityTimeout.reset();

    if (!isAsyncIterable(streamOrResponse)) {
      const move = await resolveOpenRouterMove(request, env, streamOrResponse, requestId);
      openNdjsonStream(res);
      writeStreamOpenEvent(res);
      writeNdjsonEvent(res, {
        type: 'reasoning',
        reasoning: move.reasoning,
      });
      writeNdjsonEvent(res, {
        type: 'complete',
        move,
      });
      res.end();

      if (requestId) {
        await writeRunLog({
          requestId,
          event: 'provider_succeeded',
          modelId: request.modelId,
          provider: model?.provider,
          role: request.role,
          stream: true,
          durationMs: Date.now() - startedAt,
          timeoutMs,
          sseEventCount,
          ...summarizeMove(move),
        });
      }

      return;
    }

    openNdjsonStream(res);
    writeStreamOpenEvent(res);

    for await (const chunk of streamOrResponse) {
      inactivityTimeout.reset();
      sseEventCount++;

      if (!chunk || typeof chunk !== 'object') {
        continue;
      }

      const chunkError = getOpenRouterChunkError(chunk as Record<string, unknown>);
      if (chunkError) {
        throw new Error(`OpenRouter API error: ${chunkError}`);
      }

      const delta = getOpenRouterChoiceDelta(chunk as Record<string, unknown>);
      if (!delta) {
        continue;
      }

      const reasoningDetailsDelta = getOpenRouterReasoningDetails(delta);
      if (reasoningDetailsDelta.length > 0) {
        reasoningDetails.push(...reasoningDetailsDelta);
      }

      const reasoningDelta = extractOpenRouterReasoningText(delta, '');
      if (reasoningDelta) {
        reasoningText += reasoningDelta;
        streamedText += reasoningDelta;
        writeProgressEvent(res, streamedText);
        writeNdjsonEvent(res, {
          type: 'reasoning',
          reasoning: reasoningText.trimEnd(),
        });
      }

      const outputDelta = extractOpenRouterContentText(delta);
      if (outputDelta) {
        outputText += outputDelta;
        streamedText += outputDelta;
        writeProgressEvent(res, streamedText);
      }

      if (reasoningDetailsDelta.length > 0 || outputDelta) {
        writeNdjsonEvent(res, {
          type: 'prefill',
          prefill: {
            content: outputText,
            reasoning: reasoningText.trimEnd(),
            reasoningDetails,
          },
        });
      }
    }

    const move =
      outputText.trim() ?
        await resolveMoveWithFormatRepair(request, env, {
          requestId,
          source: 'openrouter_stream',
          output: outputText,
          reasoning: reasoningText.trim(),
        })
        : await recoverOpenRouterMoveFromNonStream(request, env, requestId, 'empty_stream_output');

    if (requestId) {
      await writeRunLog({
        requestId,
        event: 'provider_succeeded',
        modelId: request.modelId,
        provider: model?.provider,
        role: request.role,
        stream: true,
        durationMs: Date.now() - startedAt,
        timeoutMs,
        sseEventCount,
        outputChars: outputText.length,
        ...summarizeMove(move),
      });
    }

    if (!reasoningText.trim()) {
      writeNdjsonEvent(res, {
        type: 'reasoning',
        reasoning: move.reasoning,
      });
    }

    writeNdjsonEvent(res, {
      type: 'complete',
      move,
    });
    res.end();
  } catch (error) {
    const resolvedError = getTimeoutSignalError(inactivityTimeout.signal, error);
    if (res.headersSent && !res.writableEnded) {
      try {
        const move = await recoverOpenRouterMoveFromNonStream(
          request,
          env,
          requestId,
          error instanceof Error ? error.message : 'stream_failed',
        );

        if (!reasoningText.trim()) {
          writeNdjsonEvent(res, {
            type: 'reasoning',
            reasoning: move.reasoning,
          });
        }

        writeNdjsonEvent(res, {
          type: 'complete',
          move,
        });
        res.end();

        if (requestId) {
          await writeRunLog({
            requestId,
            event: 'provider_succeeded',
            modelId: request.modelId,
            provider: model?.provider,
            role: request.role,
            stream: true,
            durationMs: Date.now() - startedAt,
            timeoutMs,
            sseEventCount,
            outputChars: outputText.length,
            ...summarizeMove(move),
          });
        }

        return;
      } catch {
        // Fall through to the shared stream recovery path below.
      }
    }

    if (requestId) {
      await writeRunLog({
        requestId,
        event: 'provider_failed',
        modelId: request.modelId,
        provider: model?.provider,
        role: request.role,
        stream: true,
        durationMs: Date.now() - startedAt,
        timeoutMs,
        sseEventCount,
        reasoningChars: reasoningText.length,
        outputChars: outputText.length,
        error: getClientErrorMessage(resolvedError, request.modelId, timeoutMs),
      });
    }

    throw resolvedError;
  } finally {
    inactivityTimeout.dispose();
  }
}

function splitSystemMessage(messages: Message[]) {
  const system = messages
    .filter((message) => message.role === 'system')
    .map((message) => message.content)
    .join('\n\n');

  const nonSystemMessages = messages
    .filter((message) => message.role !== 'system')
    .map((message) => ({
      role: message.role,
      content: message.content,
    }));

  return {
    system,
    messages: nonSystemMessages,
  };
}

function createAnthropicRequestInit(
  request: LlmApiRequest,
  env: Record<string, string>,
  options?: {
    stream?: boolean;
    includeThinking?: boolean;
  },
): RequestInit {
  const apiKey = pickEnvValue(env, ['ANTHROPIC_API_KEY'], /^ANTHROPIC_API_KEY(_.+)?$/);
  if (!apiKey) {
    throw new Error('Missing Anthropic API key. Set ANTHROPIC_API_KEY.');
  }

  return {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'anthropic-version': '2023-06-01',
      'x-api-key': apiKey,
    },
    body: JSON.stringify(createAnthropicRequestBody(request, options)),
  };
}

function createAnthropicRequestBody(
  request: LlmApiRequest,
  options?: {
    stream?: boolean;
    includeThinking?: boolean;
  },
) {
  const model = modelCatalogById[request.modelId];
  const { system, messages } = splitSystemMessage(request.messages);
  const thinkingConfig = options?.includeThinking ? model.anthropicThinking : undefined;
  const thinking = createAnthropicThinkingConfig(thinkingConfig);

  return {
    model: model.apiModel,
    max_tokens: 10024,
    ...(system ? { system } : {}),
    messages,
    tools: [
      {
        name: 'submit_move',
        description: 'Submit the structured Codenames move.',
        input_schema: moveSchemas[request.role],
      },
    ],
    tool_choice:
      thinking ?
        {
          type: 'auto' as const,
        }
        : {
          type: 'tool' as const,
          name: 'submit_move',
        },
    ...(thinking ? { thinking } : {}),
    ...(thinkingConfig?.effort ?
      {
        output_config: {
          effort: thinkingConfig.effort,
        },
      }
      : {}),
    ...(options?.stream ? { stream: true } : {}),
  };
}

function createAnthropicThinkingConfig(config?: AnthropicThinkingConfig) {
  if (!config) {
    return undefined;
  }

  return {
    type: config.type,
    display: config.display ?? 'summarized',
  };
}

function createOpenAiClient(modelId: string, env: Record<string, string>) {
  const apiKey = pickEnvValue(env, ['OPENAI_API_KEY'], /^OPENAI_API_KEY(_.+)?$/);
  if (!apiKey) {
    throw new Error('Missing OpenAI API key. Set OPENAI_API_KEY.');
  }

  const OpenAI = require('openai').default as typeof import('openai').default;

  return new OpenAI({
    apiKey,
    maxRetries: 0,
    timeout: getProviderTimeoutMs(modelId),
  });
}

function createGoogleGenAiClient(modelId: string, env: Record<string, string>) {
  const apiKey = pickEnvValue(
    env,
    ['GEMINI_API_KEY', 'GOOGLE_GENERATIVE_AI_API_KEY', 'GOOGLE_API_KEY'],
    /^(GEMINI|GOOGLE).*API_KEY$/,
  );
  if (!apiKey) {
    throw new Error(
      'Missing Google API key. Set GEMINI_API_KEY, GOOGLE_GENERATIVE_AI_API_KEY, or GOOGLE_API_KEY.',
    );
  }

  const { GoogleGenAI } = require(GOOGLE_GENAI_MODULE) as typeof import('@google/genai');

  return new GoogleGenAI({
    apiKey,
    apiVersion: 'v1beta',
    httpOptions: {
      timeout: getProviderTimeoutMs(modelId),
    },
  });
}

function createOpenAiResponseBody(request: LlmApiRequest) {
  const model = modelCatalogById[request.modelId];

  return {
    model: model.apiModel,
    input: request.messages.map((message) => ({
      role: message.role,
      content: [
        {
          type: 'input_text' as const,
          text: message.content,
        },
      ],
    })),
    ...(model.openAiReasoningEffort ?
      {
        reasoning: {
          effort: model.openAiReasoningEffort,
          summary: 'auto' as const,
        },
      }
      : {}),
    text: {
      format: {
        type: 'json_schema' as const,
        name: `${request.role}_move`,
        strict: true,
        schema: moveSchemas[request.role],
      },
    },
  };
}

function createGoogleGenerateContentParams(
  request: LlmApiRequest,
  options?: {
    includeThoughts?: boolean;
  },
) {
  const model = modelCatalogById[request.modelId];
  const { system, messages } = splitSystemMessage(request.messages);

  return {
    model: model.apiModel,
    contents: messages.map((message) => ({
      role: message.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: message.content }],
    })),
    config: {
      ...(system ?
        {
          systemInstruction: {
            parts: [{ text: system }],
          },
        }
        : {}),
      responseMimeType: 'application/json',
      responseJsonSchema: moveSchemas[request.role],
      ...(options?.includeThoughts ?
        {
          thinkingConfig: {
            includeThoughts: true,
          },
        }
        : {}),
    },
  };
}

function createXaiTextParams(request: LlmApiRequest, env: Record<string, string>) {
  const model = modelCatalogById[request.modelId];
  const apiKey = pickEnvValue(env, ['XAI_KEY'], /^XAI(_.+)?$/);
  if (!apiKey) {
    throw new Error('Missing xAI API key. Set XAI_KEY.');
  }

  const provider = createXai({ apiKey });
  const { system, messages } = splitSystemMessage(request.messages);

  return {
    model: provider.responses(model.apiModel),
    ...(system ? { system } : {}),
    messages,
    providerOptions: {
      xai: {
        store: false,
      },
    },
  };
}

function createOpenRouterClient(modelId: string, env: Record<string, string>) {
  const apiKey = pickEnvValue(env, ['OPENROUTER_API_KEY'], /^OPENROUTER_API_KEY(_.+)?$/);
  if (!apiKey) {
    throw new Error('Missing OpenRouter API key. Set OPENROUTER_API_KEY.');
  }

  return new OpenRouter({
    apiKey,
    httpReferer: 'https://llmcodenames.local',
    xTitle: 'LLM Codenames',
    timeoutMs: getProviderTimeoutMs(modelId),
  });
}

function createOpenRouterChatParams(
  request: LlmApiRequest,
  options?: {
    stream?: boolean;
  },
) {
  const model = modelCatalogById[request.modelId];
  const stream = options?.stream ?? false;
  const reasoningEffort = getOpenRouterReasoningEffort(model);

  return {
    model: model.apiModel,
    maxTokens: 4096,
    messages: request.messages.map((message) =>
      message.role === 'assistant' ?
        {
          role: 'assistant' as const,
          content: message.content,
          ...(message.reasoningDetails?.length ?
            {
              reasoningDetails: message.reasoningDetails,
            }
            : {}),
        }
      : {
          role: message.role,
          content: message.content,
        },
    ),
    provider: {
      allowFallbacks: true,
      ...(model.openRouterQuantizations?.length ?
        { quantizations: [...model.openRouterQuantizations] }
        : {}),
    },
    responseFormat: {
      type: 'json_object' as const,
    },
    ...(stream ? { stream: true } : {}),
    ...(!stream ? { plugins: [{ id: 'response-healing' as const }] } : {}),
    ...(reasoningEffort ? { reasoning: { effort: reasoningEffort } } : {}),
  };
}

function parseOpenAiResponsesPayload(payload: unknown) {
  if (!payload || typeof payload !== 'object') {
    throw new Error('OpenAI response payload was empty.');
  }

  const response = payload as Record<string, unknown>;
  const output = Array.isArray(response.output) ? response.output : [];

  const outputText = output
    .filter(
      (item) =>
        item &&
        typeof item === 'object' &&
        'type' in item &&
        item.type === 'message' &&
        'content' in item &&
        Array.isArray(item.content),
    )
    .flatMap((item) => (item as { content: Array<{ type?: string; text?: string }> }).content)
    .filter((part) => part.type === 'output_text' && typeof part.text === 'string')
    .map((part) => part.text as string)
    .join('\n')
    .trim();

  if (!outputText) {
    throw new Error('OpenAI response did not include output text.');
  }

  const parsedMove = parseJsonContent(outputText) as Record<string, unknown>;
  const reasoningSummary = output
    .filter(
      (item) =>
        item &&
        typeof item === 'object' &&
        'type' in item &&
        item.type === 'reasoning' &&
        'summary' in item &&
        Array.isArray(item.summary),
    )
    .flatMap((item) => (item as { summary: Array<{ type?: string; text?: string }> }).summary)
    .filter((part) => part.type === 'summary_text' && typeof part.text === 'string')
    .map((part) => part.text as string)
    .join('\n\n')
    .trim();

  if (reasoningSummary) {
    parsedMove.reasoning = reasoningSummary;
  }

  return parsedMove;
}

function parseGoogleGenerateContentPayload(payload: unknown) {
  if (!payload || typeof payload !== 'object') {
    throw new Error('Google response payload was empty.');
  }

  const outputText = getGoogleResponseText(payload).trim();
  if (!outputText) {
    throw new Error('Google response did not include output text.');
  }

  const parsedMove = parseJsonContent(outputText) as Record<string, unknown>;
  const reasoning = extractGoogleThoughtText(payload, '\n\n').trim();

  if (reasoning) {
    parsedMove.reasoning = reasoning;
  }

  return parsedMove;
}

function parseOpenRouterChatPayload(payload: unknown) {
  if (!payload || typeof payload !== 'object') {
    throw new Error('OpenRouter response payload was empty.');
  }

  const response = payload as Record<string, unknown>;
  const message = getOpenRouterChoiceMessage(response);
  const parsedMove = parseJsonContent(message?.content) as Record<string, unknown>;
  const reasoning = extractOpenRouterReasoningText(message, '\n\n');

  if (reasoning) {
    parsedMove.reasoning = reasoning;
  }

  return parsedMove;
}

async function resolveOpenRouterMove(
  request: LlmApiRequest,
  env: Record<string, string>,
  payload: unknown,
  requestId?: string,
) {
  if (!payload || typeof payload !== 'object') {
    throw new Error('OpenRouter response payload was empty.');
  }

  const response = payload as Record<string, unknown>;
  const message = getOpenRouterChoiceMessage(response);

  return resolveMoveWithFormatRepair(request, env, {
    requestId,
    source: 'openrouter_nonstream',
    output: message?.content,
    reasoning: extractOpenRouterReasoningText(message, '\n\n').trim(),
  });
}

async function resolveMoveWithFormatRepair(
  request: LlmApiRequest,
  env: Record<string, string>,
  options: {
    output: unknown;
    reasoning?: string;
    requestId?: string;
    source: string;
  },
) {
  try {
    const parsedMove = parseJsonContent(options.output) as Record<string, unknown>;
    return normalizeMove(request.role, {
      ...parsedMove,
      reasoning: options.reasoning || parsedMove.reasoning,
    });
  } catch (error) {
    return repairMoveFormatWithQwen(request, env, {
      requestId: options.requestId,
      source: options.source,
      output: options.output,
      reasoning: options.reasoning,
      originalError: error,
    });
  }
}

async function repairMoveFormatWithQwen(
  request: LlmApiRequest,
  env: Record<string, string>,
  options: {
    output: unknown;
    reasoning?: string;
    requestId?: string;
    source: string;
    originalError: unknown;
  },
) {
  const apiKey = pickEnvValue(env, ['OPENROUTER_API_KEY'], /^OPENROUTER_API_KEY(_.+)?$/);
  if (!apiKey) {
    throw options.originalError;
  }

  if (options.requestId) {
    await writeRunLog({
      requestId: options.requestId,
      event: 'format_repair_started',
      modelId: request.modelId,
      provider: modelCatalogById[request.modelId]?.provider,
      role: request.role,
      error: `${options.source}: ${getClientErrorMessage(
        options.originalError,
        request.modelId,
        getProviderTimeoutMs(request.modelId),
      )}`,
    });
  }

  const client = createOpenRouterClient(request.modelId, env);
  const repairReasoning: {
    effort?: 'medium';
  } = {
    effort: 'medium',
  };
  const response = await client.chat.send(
    {
      chatGenerationParams: {
        model: 'x-ai/grok-4.1-fast',
        maxTokens: 600,
        messages: createRepairMessages(request, options).map((message) => ({
          role: message.role,
          content: message.content,
        })),
        provider: {
          allowFallbacks: true,
        },
        responseFormat: {
          type: 'json_object' as const,
        },
        plugins: [{ id: 'response-healing' as const }],
        reasoning: repairReasoning,
      },
    },
    {
      timeoutMs: 60_000,
    },
  );

  const repairedPayload = parseOpenRouterChatPayload(response);
  const repairedMove = normalizeMove(request.role, repairedPayload);

  if (options.requestId) {
    await writeRunLog({
      requestId: options.requestId,
      event: 'format_repair_succeeded',
      modelId: request.modelId,
      provider: modelCatalogById[request.modelId]?.provider,
      role: request.role,
      ...summarizeMove(repairedMove),
    });
  }

  return repairedMove;
}

function createRepairMessages(
  request: LlmApiRequest,
  options: {
    output: unknown;
    reasoning?: string;
    source: string;
    originalError: unknown;
  },
): Message[] {
  return [
    {
      role: 'system',
      content:
        'You repair malformed model outputs for a Codenames engine. Return exactly one JSON object and nothing else. Preserve the original answer semantics. Only extract and normalize fields from the supplied output and reasoning. Do not improve the move.',
    },
    {
      role: 'user',
      content: [
        `role: ${request.role}`,
        `required_schema: ${JSON.stringify(moveSchemas[request.role])}`,
        `source: ${options.source}`,
        `original_model_id: ${request.modelId}`,
        `original_error: ${getClientErrorMessage(options.originalError, request.modelId, getProviderTimeoutMs(request.modelId))}`,
        `assistant_output:\n${serializeRepairSource(options.output) || '(empty)'}`,
        `assistant_reasoning:\n${options.reasoning?.trim() || '(empty)'}`,
        'Return only the repaired JSON object.',
      ].join('\n\n'),
    },
  ];
}

function serializeRepairSource(output: unknown) {
  if (typeof output === 'string') {
    return output.trim();
  }

  if (Array.isArray(output)) {
    const text = output
      .map((part) =>
        typeof part === 'string' ? part
          : part && typeof part === 'object' && 'text' in part && typeof part.text === 'string' ?
            part.text
            : JSON.stringify(part)
      )
      .filter(Boolean)
      .join('\n');

    return text.trim();
  }

  if (output && typeof output === 'object') {
    return JSON.stringify(output, null, 2);
  }

  return '';
}

function normalizeMove(role: MoveRole, rawMove: unknown) {
  if (!rawMove || typeof rawMove !== 'object') {
    throw new Error('Model response was not a JSON object.');
  }

  const move = rawMove as Record<string, unknown>;
  const reasoning = typeof move.reasoning === 'string' ? move.reasoning.trim() : '';
  if (!reasoning) {
    throw new Error('Model response is missing a valid reasoning string.');
  }

  if (role === 'spymaster') {
    const clue = typeof move.clue === 'string' ? move.clue.trim() : '';
    const number = normalizeInteger(move.number);

    if (!clue) {
      throw new Error('Spymaster response is missing a valid clue string.');
    }
    if (number === undefined || number < 1) {
      throw new Error('Spymaster response is missing a valid positive number.');
    }

    return {
      reasoning,
      clue,
      number,
    };
  }

  if (!Array.isArray(move.guesses)) {
    throw new Error('Operative response is missing a guesses array.');
  }

  return {
    reasoning,
    guesses: move.guesses
      .filter((guess): guess is string => typeof guess === 'string')
      .map((guess) => guess.trim())
      .filter(Boolean),
  };
}

function normalizeInteger(value: unknown) {
  if (typeof value === 'number' && Number.isInteger(value)) {
    return value;
  }

  if (typeof value === 'string' && value.trim()) {
    const parsed = Number.parseInt(value, 10);
    if (Number.isInteger(parsed)) {
      return parsed;
    }
  }

  return undefined;
}

function parseJsonContent(content: unknown) {
  if (Array.isArray(content)) {
    const text = content
      .map((part) =>
        typeof part === 'string' ? part
          : part && typeof part === 'object' && 'text' in part && typeof part.text === 'string' ?
            part.text
            : '',
      )
      .join('\n');

    return parseJsonContent(text);
  }

  if (content && typeof content === 'object') {
    return content;
  }

  if (typeof content !== 'string') {
    throw new Error('Model response content was empty.');
  }

  const candidate =
    content.includes('{') && content.includes('}') ?
      content.slice(content.indexOf('{'), content.lastIndexOf('}') + 1)
      : content;

  return JSON.parse(jsonrepair(candidate));
}

function getOpenRouterChoiceMessage(payload: Record<string, unknown>) {
  const firstChoice = Array.isArray(payload.choices) ? payload.choices[0] : undefined;
  if (!firstChoice || typeof firstChoice !== 'object' || !('message' in firstChoice)) {
    return undefined;
  }

  const { message } = firstChoice as { message?: unknown };
  return message && typeof message === 'object' ? (message as Record<string, unknown>) : undefined;
}

function getOpenRouterChoiceDelta(payload: Record<string, unknown>) {
  const firstChoice = Array.isArray(payload.choices) ? payload.choices[0] : undefined;
  if (!firstChoice || typeof firstChoice !== 'object' || !('delta' in firstChoice)) {
    return undefined;
  }

  const { delta } = firstChoice as { delta?: unknown };
  return delta && typeof delta === 'object' ? (delta as Record<string, unknown>) : undefined;
}

function extractOpenRouterReasoningText(payload: Record<string, unknown> | undefined, joiner: string) {
  if (!payload) {
    return '';
  }

  const detailsText = extractOpenRouterReasoningDetailsText(
    payload.reasoningDetails ?? payload.reasoning_details,
    joiner,
  );
  if (detailsText) {
    return detailsText;
  }

  return typeof payload.reasoning === 'string' ? payload.reasoning : '';
}

function getOpenRouterReasoningDetails(payload: Record<string, unknown> | undefined) {
  const reasoningDetails = payload?.reasoningDetails ?? payload?.reasoning_details;
  if (!Array.isArray(reasoningDetails)) {
    return [];
  }

  return reasoningDetails
    .filter((detail): detail is OpenRouterReasoningDetail => !!detail && typeof detail === 'object')
    .map((detail) => ({ ...detail }));
}

function extractGoogleThoughtText(payload: unknown, joiner: string) {
  if (!payload || typeof payload !== 'object') {
    return '';
  }

  const response = payload as Record<string, unknown>;
  const firstCandidate = Array.isArray(response.candidates) ? response.candidates[0] : undefined;
  if (!firstCandidate || typeof firstCandidate !== 'object' || !('content' in firstCandidate)) {
    return '';
  }

  const { content } = firstCandidate as { content?: unknown };
  if (!content || typeof content !== 'object' || !('parts' in content)) {
    return '';
  }

  const { parts } = content as { parts?: unknown };
  if (!Array.isArray(parts)) {
    return '';
  }

  return parts
    .filter(
      (part) =>
        part &&
        typeof part === 'object' &&
        'thought' in part &&
        part.thought === true &&
        'text' in part &&
        typeof part.text === 'string',
    )
    .map((part) => (part as { text: string }).text)
    .join(joiner);
}

function getGoogleResponseText(payload: unknown) {
  if (!payload || typeof payload !== 'object') {
    return '';
  }

  const { text } = payload as { text?: unknown };
  return typeof text === 'string' ? text : '';
}

function extractOpenRouterReasoningDetailsText(reasoningDetails: unknown, joiner: string) {
  if (!Array.isArray(reasoningDetails)) {
    return '';
  }

  return reasoningDetails
    .map((detail) => {
      if (typeof detail === 'string') {
        return detail;
      }

      if (!detail || typeof detail !== 'object') {
        return '';
      }

      if ('type' in detail && detail.type === 'encrypted') {
        return '';
      }

      if ('summary' in detail && Array.isArray(detail.summary)) {
        return detail.summary
          .map((part: unknown) =>
            part && typeof part === 'object' && 'text' in part && typeof part.text === 'string' ?
              part.text
              : '',
          )
          .filter(Boolean)
          .join(joiner);
      }

      return 'text' in detail && typeof detail.text === 'string' ? detail.text : '';
    })
    .filter(Boolean)
    .join(joiner);
}

function extractOpenRouterContentText(payload: Record<string, unknown>) {
  const { content } = payload;

  if (typeof content === 'string') {
    return content;
  }

  if (!Array.isArray(content)) {
    return '';
  }

  return content
    .map((part) =>
      typeof part === 'string' ? part
        : part && typeof part === 'object' && 'text' in part && typeof part.text === 'string' ?
          part.text
          : '',
    )
    .join('');
}

function getOpenRouterChunkError(payload: Record<string, unknown>) {
  const error = payload.error;
  if (typeof error === 'string') {
    return error;
  }

  if (error && typeof error === 'object' && 'message' in error && typeof error.message === 'string') {
    return error.message;
  }

  return '';
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return !!value && typeof value === 'object' && Symbol.asyncIterator in value;
}

async function* readSseData(
  stream: ReadableStream<Uint8Array>,
  options?: {
    onChunk?: () => void;
  },
) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (value) {
        options?.onChunk?.();
        buffer += decoder.decode(value, { stream: !done });
      }

      let eventLength = getNextSseEventLength(buffer);
      while (eventLength !== undefined) {
        const rawEvent = buffer.slice(0, eventLength);
        buffer = buffer.slice(eventLength);
        const data = extractSseData(rawEvent);
        if (data) {
          yield data;
        }
        eventLength = getNextSseEventLength(buffer);
      }

      if (done) {
        const trailingData = extractSseData(buffer);
        if (trailingData) {
          yield trailingData;
        }
        break;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function getNextSseEventLength(buffer: string) {
  const separatorMatch = buffer.match(/\r?\n\r?\n/);
  return separatorMatch ? separatorMatch.index! + separatorMatch[0].length : undefined;
}

function extractSseData(rawEvent: string) {
  const dataLines = rawEvent
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart());

  return dataLines.length > 0 ? dataLines.join('\n') : '';
}

async function readJsonBody(req: IncomingMessage) {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }

  const rawBody = Buffer.concat(chunks).toString('utf8');
  if (!rawBody) {
    throw new Error('Missing request body.');
  }

  return JSON.parse(rawBody);
}

function parseRequest(payload: unknown): LlmApiRequest {
  if (!payload || typeof payload !== 'object') {
    throw new Error('Invalid request payload.');
  }

  const request = payload as Record<string, unknown>;
  const role = request.role;
  const modelId = request.modelId;
  const messages = request.messages;

  if (role !== 'spymaster' && role !== 'operative') {
    throw new Error('Invalid role.');
  }
  if (typeof modelId !== 'string' || !modelId) {
    throw new Error('Invalid modelId.');
  }
  if (!Array.isArray(messages)) {
    throw new Error('Invalid messages payload.');
  }

  return {
    role,
    modelId,
    messages: messages.map((message) => parseMessage(message)),
    promptContext: parsePromptContext(request.promptContext),
  };
}

function parseMessage(message: unknown): Message {
  if (!message || typeof message !== 'object') {
    throw new Error('Invalid message.');
  }

  const parsed = message as Record<string, unknown>;
  if (
    (parsed.role !== 'system' && parsed.role !== 'user' && parsed.role !== 'assistant') ||
    typeof parsed.content !== 'string'
  ) {
    throw new Error('Invalid message shape.');
  }

  return {
    role: parsed.role,
    content: parsed.content,
    ...(Array.isArray(parsed.reasoningDetails) ?
      {
        reasoningDetails: parsed.reasoningDetails
          .filter((detail): detail is OpenRouterReasoningDetail => !!detail && typeof detail === 'object')
          .map((detail) => ({ ...detail })),
      }
      : {}),
  };
}

async function readApiResponse(response: Response, providerName: string) {
  const text = await response.text();
  const data = safeJsonParse(text);

  if (!response.ok) {
    throw buildApiError(response.status, text, data, providerName);
  }

  return data;
}

async function createApiError(response: Response, providerName: string) {
  const text = await response.text();
  return buildApiError(response.status, text, safeJsonParse(text), providerName);
}

function buildApiError(
  status: number,
  text: string,
  data: Record<string, unknown>,
  providerName: string,
) {
  const message =
    typeof data?.error === 'object' &&
      data.error &&
      'message' in data.error &&
      typeof data.error.message === 'string' ?
      data.error.message
      : typeof data?.error === 'string' ? data.error
        : typeof data?.message === 'string' ? data.message
          : text;

  return new Error(`${providerName} API error (${status}): ${message}`);
}

function safeJsonParse(text: string) {
  if (!text) {
    return {};
  }

  try {
    return JSON.parse(text);
  } catch {
    return {};
  }
}

function pickEnvValue(env: Record<string, string>, preferredKeys: string[], pattern: RegExp) {
  for (const key of preferredKeys) {
    if (env[key]) {
      return env[key];
    }
  }

  const fallbackKey = Object.keys(env)
    .filter((key) => pattern.test(key))
    .sort()[0];

  return fallbackKey ? env[fallbackKey] : undefined;
}

function writeJson(res: ServerResponse<IncomingMessage>, statusCode: number, body: unknown) {
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(body));
}

function openNdjsonStream(res: ServerResponse<IncomingMessage>) {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
}

function writeStreamOpenEvent(res: ServerResponse<IncomingMessage>) {
  writeNdjsonEvent(res, { type: 'stream_open' });
}

function writeNdjsonEvent(res: ServerResponse<IncomingMessage>, event: unknown) {
  res.write(`${JSON.stringify(event)}\n`);
}

function writeProgressEvent(res: ServerResponse<IncomingMessage>, streamedText: string) {
  writeNdjsonEvent(res, {
    type: 'progress',
    tokenCount: estimateTokenCount(streamedText),
  });
}

function estimateTokenCount(text: string) {
  const matches = text.match(/\w+|[^\s\w]/g);
  return matches?.length ?? 0;
}

function fetchWithTimeout(input: string, init: RequestInit, timeoutMs: number) {
  return fetch(input, {
    ...init,
    signal: AbortSignal.timeout(timeoutMs),
  });
}

function createInactivityTimeoutController(modelId: string, timeoutMs: number) {
  const controller = new AbortController();
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  const timeoutError = new DOMException(
    `Model ${modelId} timed out after ${Math.round(timeoutMs / 1000)}s of inactivity.`,
    'TimeoutError',
  );

  return {
    signal: controller.signal,
    reset() {
      if (controller.signal.aborted) {
        return;
      }

      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      timeoutId = setTimeout(() => {
        controller.abort(timeoutError);
      }, timeoutMs);
    },
    dispose() {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    },
  };
}

function getProviderTimeoutMs(modelId: string) {
  const model = modelCatalogById[modelId];
  if (!model) {
    return PROVIDER_TIMEOUT_MS.openrouter;
  }

  let timeoutMs = PROVIDER_TIMEOUT_MS[model.provider];

  if (model.provider === 'openai' && model.openAiReasoningEffort) {
    timeoutMs = Math.max(timeoutMs, REASONING_TIMEOUT_MS[model.openAiReasoningEffort]);
  }

  if (model.provider === 'openrouter' && model.openRouterReasoningEnabled) {
    timeoutMs = Math.max(timeoutMs, OPENROUTER_REASONING_TIMEOUT_MS);
  }

  return timeoutMs;
}

function getClientErrorMessage(error: unknown, modelId?: string, timeoutMs?: number) {
  if (isTimeoutError(error)) {
    if (error instanceof Error && /inactivity/i.test(error.message)) {
      return error.message;
    }

    const modelLabel = modelId ? `Model ${modelId}` : 'The model';
    const seconds = timeoutMs ? Math.round(timeoutMs / 1000) : undefined;
    return seconds ?
      `${modelLabel} timed out after ${seconds}s.`
      : `${modelLabel} timed out.`;
  }

  return error instanceof Error ? error.message : 'Unknown LLM proxy error';
}

function summarizeRequest(request: LlmApiRequest) {
  return summarizeMessages(request.messages, request.promptContext);
}

function summarizeMessages(messages: Message[], promptContext?: GamePromptContext) {
  const gameSnapshot = extractGameSnapshot(messages, promptContext);
  const summary = {
    messageCount: messages.length,
    messageChars: 0,
    systemChars: 0,
    userChars: 0,
    assistantChars: 0,
    ...(gameSnapshot ? { gameSnapshot } : {}),
  };

  for (const message of messages) {
    const contentLength = message.content.length;
    summary.messageChars += contentLength;

    if (message.role === 'system') {
      summary.systemChars += contentLength;
    } else if (message.role === 'user') {
      summary.userChars += contentLength;
    } else {
      summary.assistantChars += contentLength;
    }
  }

  return summary;
}

function extractGameSnapshot(messages: Message[], promptContext?: GamePromptContext) {
  if (promptContext) {
    const board = promptContext.board.map((card) => ({
      word: card.word,
      ...(card.color ? { color: card.color } : {}),
      isRevealed: card.isRevealed,
      ...(card.wasRecentlyRevealed ? { wasRecentlyRevealed: true } : {}),
    }));

    return {
      currentTeam: promptContext.currentTeam,
      currentRole: promptContext.currentRole,
      remainingRed: promptContext.remainingRed,
      remainingBlue: promptContext.remainingBlue,
      ...(promptContext.clue ?
        {
          clueText: promptContext.clue.text,
          clueNumber: promptContext.clue.number,
        }
        : {}),
      board,
      cardCount: board.length,
      revealedCount: board.filter((card) => card.isRevealed).length,
      recentlyRevealedCount: board.filter((card) => card.wasRecentlyRevealed).length,
    };
  }

  const userMessage = messages.find((message) => message.role === 'user')?.content;
  if (!userMessage) {
    return undefined;
  }

  const board = parseBoardSnapshot(userMessage);
  const currentTeam = readPromptLine(userMessage, 'Your Team');
  const currentRole = readPromptLine(userMessage, 'Your Role');
  const remainingRed = readPromptInteger(userMessage, 'Red Cards Left to Guess');
  const remainingBlue = readPromptInteger(userMessage, 'Blue Cards Left to Guess');
  const clueText = readPromptLine(userMessage, 'Your Clue');
  const clueNumber = readPromptInteger(userMessage, 'Number');

  if (
    !board &&
    currentTeam === undefined &&
    currentRole === undefined &&
    remainingRed === undefined &&
    remainingBlue === undefined &&
    clueText === undefined &&
    clueNumber === undefined
  ) {
    return undefined;
  }

  return {
    ...(currentTeam ? { currentTeam } : {}),
    ...(currentRole ? { currentRole } : {}),
    ...(remainingRed !== undefined ? { remainingRed } : {}),
    ...(remainingBlue !== undefined ? { remainingBlue } : {}),
    ...(clueText ? { clueText } : {}),
    ...(clueNumber !== undefined ? { clueNumber } : {}),
    ...(board ?
      {
        board,
        cardCount: board.length,
        revealedCount: board.filter((card) => card.isRevealed).length,
        recentlyRevealedCount: board.filter((card) => card.wasRecentlyRevealed).length,
      }
      : {}),
  };
}

function parseBoardSnapshot(userMessage: string) {
  const boardMarker = 'Board:';
  const boardStart = userMessage.indexOf(boardMarker);
  if (boardStart === -1) {
    return undefined;
  }

  const boardWithTrailingPrompt = userMessage.slice(boardStart + boardMarker.length);
  const clueMarkerIndex = boardWithTrailingPrompt.indexOf('\n\n  Your Clue:');
  const boardText =
    clueMarkerIndex === -1 ? boardWithTrailingPrompt.trim() : boardWithTrailingPrompt.slice(0, clueMarkerIndex).trim();

  try {
    const parsed = JSON.parse(boardText);
    if (!Array.isArray(parsed)) {
      return undefined;
    }

    return parsed
      .filter((card): card is Record<string, unknown> => Boolean(card) && typeof card === 'object')
      .map((card) => ({
        word: typeof card.word === 'string' ? card.word : '',
        ...(typeof card.color === 'string' ? { color: card.color } : {}),
        ...(typeof card.isRevealed === 'boolean' ? { isRevealed: card.isRevealed } : {}),
        ...(typeof card.wasRecentlyRevealed === 'boolean' ?
          { wasRecentlyRevealed: card.wasRecentlyRevealed }
          : {}),
      }))
      .filter((card) => card.word);
  } catch {
    return undefined;
  }
}

function readPromptLine(userMessage: string, label: string) {
  const match = userMessage.match(new RegExp(`^${escapeRegExp(label)}:\\s*(.+)$`, 'm'));
  return match?.[1]?.trim() || undefined;
}

function readPromptInteger(userMessage: string, label: string) {
  const rawValue = readPromptLine(userMessage, label);
  if (!rawValue) {
    return undefined;
  }

  const parsed = Number.parseInt(rawValue, 10);
  return Number.isInteger(parsed) ? parsed : undefined;
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function summarizeRawResult(rawResult: unknown) {
  if (!rawResult || typeof rawResult !== 'object') {
    return {};
  }

  const result = rawResult as Record<string, unknown>;
  const summary: Omit<RequestLogEntry, 'timestamp' | 'requestId' | 'event'> = {
    responseKeys: Object.keys(result).sort(),
  };

  if (typeof result.reasoning === 'string') {
    summary.reasoningChars = result.reasoning.trim().length;
  }
  if (typeof result.clue === 'string') {
    summary.clue = result.clue.trim();
  }
  if (typeof result.number === 'number' && Number.isInteger(result.number)) {
    summary.number = result.number;
  }
  if (Array.isArray(result.guesses)) {
    const guesses = result.guesses.filter((guess): guess is string => typeof guess === 'string');
    summary.guessCount = guesses.length;
    summary.guesses = guesses;
  }

  return summary;
}

function summarizeMove(move: ReturnType<typeof normalizeMove>) {
  const summary: Omit<RequestLogEntry, 'timestamp' | 'requestId' | 'event'> = {
    reasoningChars: move.reasoning.length,
  };

  if ('clue' in move) {
    summary.clue = move.clue;
    summary.number = move.number;
  } else {
    summary.guessCount = move.guesses.length;
    summary.guesses = move.guesses;
  }

  return summary;
}

function isTimeoutError(error: unknown) {
  return (
    (error instanceof DOMException && error.name === 'TimeoutError') ||
    (error instanceof Error && error.name === 'TimeoutError')
  );
}

function getTimeoutSignalError(signal: AbortSignal, error: unknown) {
  return signal.aborted && isTimeoutError(signal.reason) ? signal.reason : error;
}

async function writeRunLog(entry: RequestLogEntry) {
  try {
    if (!logDirectoryReady) {
      logDirectoryReady = mkdir(LOG_DIR_URL, { recursive: true }).then(() => undefined);
    }

    await logDirectoryReady;
    await appendFile(
      RUN_LOG_URL,
      `${JSON.stringify({
        ...entry,
        timestamp: new Date().toISOString(),
      })}\n`,
      'utf8',
    );
  } catch (error) {
    console.error('[llm-proxy] failed to write run log:', error);
  }
}

async function writeAnthropicTranscript(payload: {
  requestId?: string;
  modelId: string;
  role: MoveRole;
  startedAt: string;
  finishedAt: string;
  status: 'succeeded' | 'failed';
  stopReason?: string;
  error?: string;
  messageSummary: ReturnType<typeof summarizeMessages>;
  reasoningText: string;
  assistantText: string;
  streamedToolInput: Record<string, unknown> | null;
  move?: ReturnType<typeof normalizeMove>;
  events: AnthropicTranscriptEvent[];
}) {
  if (!payload.requestId) {
    return;
  }

  try {
    if (!anthropicTranscriptDirectoryReady) {
      anthropicTranscriptDirectoryReady = mkdir(ANTHROPIC_TRANSCRIPT_DIR_URL, {
        recursive: true,
      }).then(() => undefined);
    }

    await anthropicTranscriptDirectoryReady;
    await writeFile(
      new URL(`${payload.requestId}.json`, ANTHROPIC_TRANSCRIPT_DIR_URL),
      JSON.stringify(payload, null, 2),
      'utf8',
    );
  } catch (error) {
    console.error('[llm-proxy] failed to write Anthropic transcript:', error);
  }
}
