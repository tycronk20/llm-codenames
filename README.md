# LLM Codenames

![LLM Codenames Screenshot](/public/codenames-screenshot.png)

An implementation of the board game [Codenames](<https://en.wikipedia.org/wiki/Codenames_(board_game)>), with all four players replaced with frontier LLM agents.

## What Changed

- Removed the Cloudflare Worker dependency
- Replaced the OpenRouter-only model list with a mixed frontier roster defined in `src/utils/modelCatalog.ts`
- Added a server-side `/api/llm` route inside the Vite dev/preview server so API keys stay out of the browser bundle

## Provider Routing

Provider-to-model routing is defined in `src/utils/modelCatalog.ts`. The active UI roster comes from `activeModelIds`, while the full supported set lives in `allModelCatalog`.

Notes:
- Google currently exposes the requested Gemini models through the `gemini-3-pro-preview` and `gemini-3-flash-preview` API IDs.
- In this repo, the benchmark entry is named `Gemini 3.1 Pro` and keyed as `gemini-3.1-pro`, but it currently calls Google's `gemini-3-pro-preview` API model.
- OpenRouter reasoning support is configured per model in `src/utils/modelCatalog.ts` via `openRouterReasoningEnabled`.
- The complete supported-model catalog lives in `src/utils/modelCatalog.ts` as `allModelCatalog`; the active roster is derived separately from `activeModelIds`.
- Active model availability is controlled by `activeModelIds` in `src/utils/modelCatalog.ts`. Keep at least 4 models enabled so the game can assign one unique model to each role.

## Environment

Create a local `.env` from `.env.example` and set:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`

The Vite middleware also accepts common fallbacks already seen in local shells:

- `ANTHROPIC_API_KEY_*`
- `GOOGLE_GENERATIVE_AI_API_KEY`
- `GOOGLE_API_KEY`

## Installation

- `bun install`
- `cp .env.example .env`
- `bun run dev`

The browser app calls `/api/llm`, and the Vite server performs the provider requests server-side.

## Build

- `bun run build`
- `bunx vite preview`

The `/api/llm` route is available in `dev` and `preview`. If you deploy the static `dist/` output somewhere else, you will need an equivalent server or serverless route for the LLM calls.
