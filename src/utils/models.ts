import anthropicLogo from '../assets/logos/anthropic.svg';
import deepseekLogo from '../assets/logos/deepseek.svg';
import geminiLogo from '../assets/logos/gemini.svg';
import openaiLogo from '../assets/logos/openai.svg';
import openrouterLogo from '../assets/logos/openrouter.svg';
import xaiLogo from '../assets/logos/xai.svg';
import {
  allModelCatalog,
  LLMModelConfig,
  LogoKey,
  MIN_ACTIVE_MODELS,
  modelCatalog,
} from './modelCatalog';

export type LLMModel = LLMModelConfig & {
  logo: string;
};

const ACTIVE_MODEL_IDS_STORAGE_KEY = 'llm-codenames:active-model-ids:v1';
export const ACTIVE_MODEL_IDS_UPDATED_EVENT = 'llm-codenames:active-model-ids-updated';

function canUseBrowserStorage() {
  return typeof globalThis.localStorage !== 'undefined';
}

function canDispatchBrowserEvents() {
  return typeof globalThis.dispatchEvent === 'function' && typeof globalThis.CustomEvent === 'function';
}

const logoByKey: Record<LogoKey, string> = {
  anthropic: anthropicLogo,
  deepseek: deepseekLogo,
  gemini: geminiLogo,
  openai: openaiLogo,
  openrouter: openrouterLogo,
  xai: xaiLogo,
};

function withLogo(model: LLMModelConfig): LLMModel {
  return {
    ...model,
    logo: logoByKey[model.logoKey],
  };
}

export const allModels: LLMModel[] = allModelCatalog.map(withLogo);

export const allModelsById: Record<string, LLMModel> = Object.fromEntries(
  allModels.map((model) => [model.id, model]),
);

export const agents: LLMModel[] = modelCatalog.map(withLogo);

export const agentsById: Record<string, LLMModel> = Object.fromEntries(
  agents.map((agent) => [agent.id, agent]),
);

export function getStoredActiveModelIds() {
  if (!canUseBrowserStorage()) {
    return null;
  }

  try {
    const rawValue = globalThis.localStorage.getItem(ACTIVE_MODEL_IDS_STORAGE_KEY);
    if (!rawValue) {
      return null;
    }

    const parsed = JSON.parse(rawValue);
    if (!Array.isArray(parsed)) {
      return null;
    }

    const validIds = parsed.filter(
      (id): id is string => typeof id === 'string' && Boolean(allModelsById[id]),
    );

    return validIds.length >= MIN_ACTIVE_MODELS ? validIds : null;
  } catch {
    return null;
  }
}

export function getActiveModelIds() {
  return getStoredActiveModelIds() ?? agents.map((agent) => agent.id);
}

export function getActiveModels() {
  return getActiveModelIds()
    .map((modelId) => allModelsById[modelId])
    .filter((model): model is LLMModel => Boolean(model));
}

export function saveActiveModelIds(modelIds: string[]) {
  const uniqueValidIds = Array.from(new Set(modelIds)).filter((id) => Boolean(allModelsById[id]));
  if (uniqueValidIds.length < MIN_ACTIVE_MODELS) {
    throw new Error(`At least ${MIN_ACTIVE_MODELS} models must be active.`);
  }

  if (!canUseBrowserStorage()) {
    return uniqueValidIds;
  }

  globalThis.localStorage.setItem(ACTIVE_MODEL_IDS_STORAGE_KEY, JSON.stringify(uniqueValidIds));

  if (canDispatchBrowserEvents()) {
    globalThis.dispatchEvent(
      new CustomEvent(ACTIVE_MODEL_IDS_UPDATED_EVENT, {
        detail: uniqueValidIds,
      }),
    );
  }

  return uniqueValidIds;
}

export function resolveModel(modelId: string | undefined) {
  if (!modelId) {
    return null;
  }

  return allModelsById[modelId] ?? null;
}
