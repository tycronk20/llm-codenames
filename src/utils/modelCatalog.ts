export type Provider = 'google' | 'openai' | 'anthropic' | 'openrouter';

export type LogoKey = 'anthropic' | 'deepseek' | 'gemini' | 'openai' | 'openrouter' | 'xai';

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
  anthropicThinking?: {
    type: 'adaptive';
    display?: 'summarized' | 'omitted';
    effort?: 'low' | 'medium' | 'high' | 'max';
  };
};

export const allModelCatalog = [
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
    id: 'deepseek-v3.2-reasoning',
    provider: 'openrouter',
    apiModel: 'deepseek/deepseek-v3.2',
    modelName: 'DeepSeek V3.2 Reasoning',
    shortName: 'DeepSeek V3.2 R',
    logoKey: 'deepseek',
    openRouterReasoningEnabled: true,
    openRouterReasoningEffort: 'medium',
  },
  {
    id: 'deepseek-v3.2',
    provider: 'openrouter',
    apiModel: 'deepseek/deepseek-v3.2',
    modelName: 'DeepSeek V3.2',
    shortName: 'DeepSeek V3.2',
    logoKey: 'deepseek',
    openRouterReasoningEnabled: false,
    openRouterReasoningEffort: 'none',
  },
  {
    id: 'grok-4.1-fast',
    provider: 'openrouter',
    apiModel: 'x-ai/grok-4.1-fast',
    modelName: 'Grok 4.1 Fast',
    shortName: 'Grok 4.1 Fast',
    logoKey: 'xai',
    autoResumeOnIdle: false,
    openRouterReasoningEnabled: true,
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
    provider: 'openrouter',
    apiModel: 'x-ai/grok-4.20-beta',
    modelName: 'Grok 4.20 Beta',
    shortName: 'Grok 4.20',
    logoKey: 'xai',
    openRouterReasoningEnabled: true,
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

type ModelId = (typeof allModelCatalog)[number]['id'];

export const activeModelIds: readonly ModelId[] = [
  'gemini-3-flash',
  'grok-4.1-fast',
  'qwen3.5-27b-reasoning',
  'deepseek-v3.2-reasoning',
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
