import anthropicLogo from '../assets/logos/anthropic.svg';
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
  gemini: geminiLogo,
  openai: openaiLogo,
  openrouter: openrouterLogo,
  xai: xaiLogo,
};

export const agents: LLMModel[] = modelCatalog.map((model) => ({
  ...model,
  logo: logoByKey[model.logoKey],
}));
