export type PromptBoardCard = {
  word: string;
  isRevealed: boolean;
  color?: string;
  wasRecentlyRevealed?: boolean;
};

export type PromptSourceState = {
  currentTeam: 'red' | 'blue';
  currentRole: 'spymaster' | 'operative';
  remainingRed: number;
  remainingBlue: number;
  cards: PromptBoardCard[];
  currentClue?: {
    clueText: string;
    number: number;
  };
};

export type GamePromptContext = {
  currentTeam: PromptSourceState['currentTeam'];
  currentRole: PromptSourceState['currentRole'];
  remainingRed: number;
  remainingBlue: number;
  board: PromptBoardCard[];
  clue?: {
    text: string;
    number: number;
  };
};

export function createPromptContext(sourceState: PromptSourceState): GamePromptContext {
  return {
    currentTeam: sourceState.currentTeam,
    currentRole: sourceState.currentRole,
    remainingRed: sourceState.remainingRed,
    remainingBlue: sourceState.remainingBlue,
    board:
      sourceState.currentRole === 'spymaster' ?
        sourceState.cards.map((card) => ({
          word: card.word,
          color: card.color,
          isRevealed: card.isRevealed,
          wasRecentlyRevealed: card.wasRecentlyRevealed,
        }))
      : sourceState.cards.map((card) => ({
          word: card.word,
          isRevealed: card.isRevealed,
          ...(card.isRevealed ? { color: card.color } : {}),
          ...(card.wasRecentlyRevealed ? { wasRecentlyRevealed: true } : {}),
        })),
    ...(sourceState.currentRole === 'operative' && sourceState.currentClue ?
      {
        clue: {
          text: sourceState.currentClue.clueText,
          number: sourceState.currentClue.number,
        },
      }
      : {}),
  };
}

export function parsePromptContext(payload: unknown): GamePromptContext | undefined {
  if (!payload || typeof payload !== 'object') {
    return undefined;
  }

  const promptContext = payload as Record<string, unknown>;
  if (
    (promptContext.currentTeam !== 'red' && promptContext.currentTeam !== 'blue') ||
    (promptContext.currentRole !== 'spymaster' && promptContext.currentRole !== 'operative') ||
    typeof promptContext.remainingRed !== 'number' ||
    typeof promptContext.remainingBlue !== 'number' ||
    !Array.isArray(promptContext.board)
  ) {
    return undefined;
  }

  const board = promptContext.board
    .filter((card): card is Record<string, unknown> => Boolean(card) && typeof card === 'object')
    .map((card) => ({
      word: typeof card.word === 'string' ? card.word : '',
      isRevealed: Boolean(card.isRevealed),
      ...(typeof card.color === 'string' ? { color: card.color } : {}),
      ...(card.wasRecentlyRevealed ? { wasRecentlyRevealed: true } : {}),
    }))
    .filter((card) => card.word);

  const clue =
    promptContext.clue &&
    typeof promptContext.clue === 'object' &&
    typeof (promptContext.clue as Record<string, unknown>).text === 'string' &&
    typeof (promptContext.clue as Record<string, unknown>).number === 'number' ?
      {
        text: (promptContext.clue as Record<string, unknown>).text as string,
        number: (promptContext.clue as Record<string, unknown>).number as number,
      }
    : undefined;

  return {
    currentTeam: promptContext.currentTeam,
    currentRole: promptContext.currentRole,
    remainingRed: promptContext.remainingRed,
    remainingBlue: promptContext.remainingBlue,
    board,
    ...(clue ? { clue } : {}),
  };
}

export function renderUserPrompt(promptContext: GamePromptContext): string {
  return `
### Current Game State
Your Team: ${promptContext.currentTeam}
Your Role: ${promptContext.currentRole}
Red Cards Left to Guess: ${promptContext.remainingRed}
Blue Cards Left to Guess: ${promptContext.remainingBlue}

Board: ${JSON.stringify(promptContext.board)}

${
  promptContext.clue ?
    `
  Your Clue: ${promptContext.clue.text}
  Number: ${promptContext.clue.number}
  `
  : ''
}
`;
}

export const createUserPrompt = (gameState: PromptSourceState): string =>
  renderUserPrompt(createPromptContext(gameState));

