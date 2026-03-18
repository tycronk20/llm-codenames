import anthropicLogo from '../assets/logos/anthropic.svg';
import deepseekLogo from '../assets/logos/deepseek.svg';
import geminiLogo from '../assets/logos/gemini.svg';
import openaiLogo from '../assets/logos/openai.svg';
import openrouterLogo from '../assets/logos/openrouter.svg';
import xaiLogo from '../assets/logos/xai.svg';
import { LLMModelConfig, LogoKey, modelCatalog } from './modelCatalog';

export type LLMModel = LLMModelConfig & {
  logo: string;
};

const logoByKey: Record<LogoKey, string> = {
  anthropic: anthropicLogo,
  deepseek: deepseekLogo,
  gemini: geminiLogo,
  openai: openaiLogo,
  openrouter: openrouterLogo,
  xai: xaiLogo,
};

export const agents: LLMModel[] = modelCatalog.map((model) => ({
  ...model,
  logo: logoByKey[model.logoKey],
}));

export const agentsById: Record<string, LLMModel> = Object.fromEntries(
  agents.map((agent) => [agent.id, agent]),
);
