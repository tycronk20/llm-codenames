export type Provider = 'google' | 'openai' | 'anthropic' | 'openrouter';

export type LogoKey = 'anthropic' | 'gemini' | 'openai' | 'openrouter' | 'xai';

export type LLMModelConfig = {
  id: string;
  provider: Provider;
  apiModel: string;
  modelName: string;
  shortName: string;
  logoKey: LogoKey;
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
    provider: 'google',
    apiModel: 'gemini-3-pro-preview',
    modelName: 'Gemini 3.1 Pro',
    shortName: 'Gemini 3.1 Pro',
    logoKey: 'gemini',
  },
  {
    id: 'gemini-3-flash',
    provider: 'google',
    apiModel: 'gemini-3-flash-preview',
    modelName: 'Gemini 3 Flash',
    shortName: 'Gemini 3 Flash',
    logoKey: 'gemini',
  },
  {
    id: 'gpt-5.4-medium',
    provider: 'openai',
    apiModel: 'gpt-5.4',
    modelName: 'GPT-5.4 Medium',
    shortName: 'GPT-5.4 Medium',
    logoKey: 'openai',
    openAiReasoningEffort: 'medium',
  },
  {
    id: 'gpt-5.4-high',
    provider: 'openai',
    apiModel: 'gpt-5.4',
    modelName: 'GPT-5.4 High',
    shortName: 'GPT-5.4 High',
    logoKey: 'openai',
    openAiReasoningEffort: 'high',
  },
  {
    id: 'gpt-5.4-xhigh',
    provider: 'openai',
    apiModel: 'gpt-5.4',
    modelName: 'GPT-5.4 XHigh',
    shortName: 'GPT-5.4 XHigh',
    logoKey: 'openai',
    openAiReasoningEffort: 'xhigh',
  },
  {
    id: 'claude-opus-4.6',
    provider: 'anthropic',
    apiModel: 'claude-opus-4-6',
    modelName: 'Claude Opus 4.6',
    shortName: 'Opus 4.6',
    logoKey: 'anthropic',
    anthropicThinking: {
      type: 'adaptive',
      display: 'summarized',
      effort: 'high',
    },
  },
  {
    id: 'claude-sonnet-4.6',
    provider: 'anthropic',
    apiModel: 'claude-sonnet-4-6',
    modelName: 'Claude Sonnet 4.6',
    shortName: 'Sonnet 4.6',
    logoKey: 'anthropic',
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
    id: 'grok-4.1-fast',
    provider: 'openrouter',
    apiModel: 'x-ai/grok-4.1-fast',
    modelName: 'Grok 4.1 Fast',
    shortName: 'Grok 4.1 Fast',
    logoKey: 'xai',
    openRouterReasoningEnabled: true,
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
  },
  {
    id: 'gpt-5.3-chat',
    provider: 'openai',
    apiModel: 'gpt-5.3-chat-latest',
    modelName: 'GPT-5.3 Chat',
    shortName: 'GPT-5.3 Chat',
    logoKey: 'openai',
    openAiReasoningEffort: 'medium',
  },
 ] as const satisfies readonly LLMModelConfig[];

type ModelId = (typeof allModelCatalog)[number]['id'];

export const activeModelIds: readonly ModelId[] = [
  'gemini-3-flash',
  'kimi-2.5-thinking',
  'grok-4.20-beta',
  'glm-5',
  'gpt-5.3-chat',
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

export const modelCatalogById = Object.fromEntries(
  modelCatalog.map((model) => [model.id, model]),
) satisfies Record<string, LLMModelConfig>;
