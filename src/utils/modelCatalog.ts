export type Provider = 'google' | 'openai' | 'anthropic' | 'openrouter' | 'xai';

export type LogoKey = 'anthropic' | 'deepseek' | 'gemini' | 'openai' | 'openrouter' | 'xai';

/** OpenRouter `provider.quantizations` routing filter (matches OpenRouter API). */
export type OpenRouterQuantization =
  | 'int4'
  | 'int8'
  | 'fp4'
  | 'fp6'
  | 'fp8'
  | 'fp16'
  | 'bf16'
  | 'fp32'
  | 'unknown';

export type LLMModelConfig = {
  id: string;
  provider: Provider;
  apiModel: string;
  modelName: string;
  shortName: string;
  logoKey: LogoKey;
  fallbackProvider?: Exclude<Provider, 'openrouter'>;
  fallbackApiModel?: string;
  openRouterAssistantPrefillEnabled?: boolean;
  autoResumeOnIdle?: boolean;
  openRouterReasoningEffort?: 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'none';
  openAiReasoningEffort?: 'medium' | 'high' | 'xhigh';
  openRouterReasoningEnabled?: boolean;
  /** When set, only OpenRouter endpoints matching these quantization levels are eligible. */
  openRouterQuantizations?: readonly OpenRouterQuantization[];
  anthropicThinking?: {
    type: 'adaptive';
    display?: 'summarized' | 'omitted';
    effort?: 'low' | 'medium' | 'high' | 'max';
  };
};

import { additionalModelCatalog } from './modelCatalogExtras.ts';

export const MIN_ACTIVE_MODELS = 4;

const baseModelCatalog = [
  {
    id: 'gemini-3.1-pro',
    provider: 'openrouter',
    apiModel: 'google/gemini-3.1-pro-preview',
    modelName: 'Gemini 3.1 Pro',
    shortName: 'Gemini 3.1 Pro',
    logoKey: 'gemini',
    fallbackProvider: 'google',
    fallbackApiModel: 'gemini-3-pro-preview',
    openRouterReasoningEnabled: true,
  },
  {
    id: 'gemini-3-flash',
    provider: 'openrouter',
    apiModel: 'google/gemini-3-flash-preview',
    modelName: 'Gemini 3 Flash',
    shortName: 'Gemini 3 Flash',
    logoKey: 'gemini',
    fallbackProvider: 'google',
    fallbackApiModel: 'gemini-3-flash-preview',
    openRouterReasoningEnabled: true,
  },
  {
    id: 'gpt-5.4-medium',
    provider: 'openrouter',
    apiModel: 'openai/gpt-5.4',
    modelName: 'GPT-5.4 Medium',
    shortName: 'GPT-5.4 Medium',
    logoKey: 'openai',
    fallbackProvider: 'openai',
    fallbackApiModel: 'gpt-5.4',
    openAiReasoningEffort: 'medium',
    openRouterReasoningEnabled: true,
  },
  {
    id: 'gpt-5.4-high',
    provider: 'openrouter',
    apiModel: 'openai/gpt-5.4',
    modelName: 'GPT-5.4 High',
    shortName: 'GPT-5.4 High',
    logoKey: 'openai',
    fallbackProvider: 'openai',
    fallbackApiModel: 'gpt-5.4',
    openAiReasoningEffort: 'high',
    openRouterReasoningEnabled: true,
  },
  {
    id: 'gpt-5.4-xhigh',
    provider: 'openrouter',
    apiModel: 'openai/gpt-5.4',
    modelName: 'GPT-5.4 XHigh',
    shortName: 'GPT-5.4 XHigh',
    logoKey: 'openai',
    fallbackProvider: 'openai',
    fallbackApiModel: 'gpt-5.4',
    openAiReasoningEffort: 'xhigh',
    openRouterReasoningEnabled: true,
  },
  {
    id: 'claude-opus-4.6',
    provider: 'openrouter',
    apiModel: 'anthropic/claude-opus-4.6',
    modelName: 'Claude Opus 4.6',
    shortName: 'Opus 4.6',
    logoKey: 'anthropic',
    fallbackProvider: 'anthropic',
    fallbackApiModel: 'claude-opus-4-6',
    openRouterAssistantPrefillEnabled: false,
    openRouterReasoningEnabled: true,
    anthropicThinking: {
      type: 'adaptive',
      display: 'summarized',
      effort: 'high',
    },
  },
  {
    id: 'claude-sonnet-4.6',
    provider: 'openrouter',
    apiModel: 'anthropic/claude-sonnet-4.6',
    modelName: 'Claude Sonnet 4.6',
    shortName: 'Sonnet 4.6',
    logoKey: 'anthropic',
    fallbackProvider: 'anthropic',
    fallbackApiModel: 'claude-sonnet-4-6',
    openRouterAssistantPrefillEnabled: false,
    openRouterReasoningEnabled: true,
    anthropicThinking: {
      type: 'adaptive',
      display: 'summarized',
      effort: 'high',
    },
  },
  {
    id: 'kimi-2.5-thinking',
    provider: 'openrouter',
    apiModel: 'moonshotai/kimi-k2.5',
    modelName: 'Kimi 2.5 Thinking',
    shortName: 'Kimi 2.5',
    logoKey: 'openrouter',
    openRouterReasoningEnabled: true,
  },
  {
    id: 'deepseek-v3-2-reasoning',
    provider: 'openrouter',
    apiModel: 'deepseek/deepseek-v3.2',
    modelName: 'DeepSeek V3.2 Reasoning',
    shortName: 'DeepSeek V3.2 R',
    logoKey: 'deepseek',
    openRouterAssistantPrefillEnabled: false,
    autoResumeOnIdle: false,
    openRouterReasoningEnabled: true,
    openRouterReasoningEffort: 'medium',
    openRouterQuantizations: ['fp8', 'fp16'],
  },
  {
    id: 'deepseek-v3-2',
    provider: 'openrouter',
    apiModel: 'deepseek/deepseek-v3.2',
    modelName: 'DeepSeek V3.2',
    shortName: 'DeepSeek V3.2',
    logoKey: 'deepseek',
    openRouterReasoningEnabled: false,
    openRouterReasoningEffort: 'none',
    openRouterQuantizations: ['fp8', 'fp16'],
  },
  {
    id: 'grok-4.1-fast',
    provider: 'xai',
    apiModel: 'grok-4-1-fast-reasoning',
    modelName: 'Grok 4.1 Fast',
    shortName: 'Grok 4.1 Fast',
    logoKey: 'xai',
    autoResumeOnIdle: false,
  },
  {
    id: 'gpt-oss-120b-high',
    provider: 'openrouter',
    apiModel: 'openai/gpt-oss-120b',
    modelName: 'GPT-OSS 120B High',
    shortName: 'GPT-OSS 120B High',
    logoKey: 'openai',
    openRouterReasoningEnabled: true,
    openRouterReasoningEffort: 'high',
  },
  {
    id: 'qwen3.5-27b-reasoning',
    provider: 'openrouter',
    apiModel: 'qwen/qwen3.5-27b',
    modelName: 'Qwen3.5 27B Reasoning',
    shortName: 'Qwen3.5 27B',
    logoKey: 'openrouter',
    openRouterReasoningEnabled: true,
  },
  {
    id: 'mimo-v2-flash',
    provider: 'openrouter',
    apiModel: 'xiaomi/mimo-v2-flash',
    modelName: 'MiMo-V2 Flash',
    shortName: 'MiMo-V2 Flash',
    logoKey: 'openrouter',
  },
  {
    id: 'grok-4.20-beta',
    provider: 'xai',
    apiModel: 'grok-4.20-0309-reasoning',
    modelName: 'Grok 4.20 Beta',
    shortName: 'Grok 4.20',
    logoKey: 'xai',
  },
  {
    id: 'glm-5',
    provider: 'openrouter',
    apiModel: 'z-ai/glm-5',
    modelName: 'GLM-5',
    shortName: 'GLM-5',
    logoKey: 'openrouter',
    openRouterReasoningEnabled: true,
  },
  {
    id: 'gpt-5.3-chat',
    provider: 'openrouter',
    apiModel: 'openai/gpt-5.3-chat',
    modelName: 'GPT-5.3 Chat',
    shortName: 'GPT-5.3 Chat',
    logoKey: 'openai',
    fallbackProvider: 'openai',
    fallbackApiModel: 'gpt-5.3-chat-latest',
    openAiReasoningEffort: 'medium',
    openRouterReasoningEnabled: true,
  },
] as const satisfies readonly LLMModelConfig[];

export const allModelCatalog = [...baseModelCatalog, ...additionalModelCatalog] as const satisfies readonly LLMModelConfig[];

type ModelId = (typeof allModelCatalog)[number]['id'];

export const activeModelIds: readonly ModelId[] = [
  'gemini-3-flash',
  'grok-4.1-fast',
  'qwen3.5-27b-reasoning',
  'deepseek-v3-2-reasoning',
];

export const allModelCatalogById = Object.fromEntries(
  allModelCatalog.map((model) => [model.id, model]),
) as Record<ModelId, LLMModelConfig>;

export const modelCatalog: LLMModelConfig[] = activeModelIds.map((modelId) => {
  const model = allModelCatalogById[modelId];
  if (!model) {
    throw new Error(`Unknown active model id: ${modelId}`);
  }

  return model;
});

export const modelCatalogById: Record<string, LLMModelConfig> = allModelCatalogById;

export function isAssistantPrefillEnabled(
  model: Pick<LLMModelConfig, 'openRouterAssistantPrefillEnabled'>,
) {
  return model.openRouterAssistantPrefillEnabled !== false;
}

export function isAutoResumeOnIdleEnabled(model: Pick<LLMModelConfig, 'autoResumeOnIdle'>) {
  return model.autoResumeOnIdle !== false;
}

export function getOpenRouterReasoningEffort(
  model: Pick<LLMModelConfig, 'openRouterReasoningEffort' | 'openRouterReasoningEnabled'>,
) {
  if (model.openRouterReasoningEffort && model.openRouterReasoningEffort !== 'none') {
    return model.openRouterReasoningEffort;
  }

  return model.openRouterReasoningEnabled ? 'medium' : undefined;
}

export function hasOpenRouterReasoning(
  model: Pick<LLMModelConfig, 'openRouterReasoningEffort' | 'openRouterReasoningEnabled'>,
) {
  return getOpenRouterReasoningEffort(model) !== undefined;
}
