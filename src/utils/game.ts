import wordlist from '../assets/wordlist-eng.txt?raw';
import { LLMModel } from './models';
import { MIN_ACTIVE_MODELS } from './modelCatalog';
// Core types
export type TeamColor = 'red' | 'blue';
export type CardColor = 'red' | 'blue' | 'black' | 'neutral';
export type Role = 'spymaster' | 'operative';

export type CardType = {
  word: string;
  color: CardColor;
  isRevealed: boolean;
  wasRecentlyRevealed: boolean;
};

export type SpymasterMove = {
  clue: string;
  number: number;
  reasoning: string;
};

export type OperativeMove = {
  guesses: string[];
  reasoning: string;
};

export function formatMoveMessage(role: 'spymaster', move: SpymasterMove): string;
export function formatMoveMessage(role: 'operative', move: OperativeMove): string;
export function formatMoveMessage(
  role: Role,
  move: SpymasterMove | OperativeMove,
): string {
  if (role === 'spymaster') {
    const spymasterMove = move as SpymasterMove;
    return `${spymasterMove.reasoning}\n\nClue: ${spymasterMove.clue.toUpperCase()}, ${spymasterMove.number}`;
  }

  const operativeMove = move as OperativeMove;
  return `${operativeMove.reasoning}\n\nGuesses: ${operativeMove.guesses.join(', ')}`;
}

export type ChatMessage = {
  model: LLMModel;
  message: string;
  team: TeamColor;
  cards?: CardType[];
  isStreaming?: boolean;
};

export type TeamAgents = {
  [key in Role]: LLMModel;
};

export type GameAgents = {
  [key in TeamColor]: TeamAgents;
};

// Game state
export type GameState = {
  agents: GameAgents;
  cards: CardType[];
  chatHistory: ChatMessage[];
  currentTeam: TeamColor;
  currentRole: Role;
  previousTeam?: TeamColor;
  previousRole?: Role;
  remainingRed: number;
  remainingBlue: number;
  currentClue?: {
    clueText: string;
    number: number;
  };
  currentGuesses?: string[];
  gameWinner?: TeamColor;
};

// Initialize new game state
export function initializeGameState(activeModels: LLMModel[] = []): GameState {
  return {
    cards: drawNewCards(),
    agents: selectRandomAgents(activeModels),
    currentTeam: 'red',
    currentRole: 'spymaster',
    remainingRed: 9,
    remainingBlue: 8,
    chatHistory: [],
  };
}

const drawNewCards = (): CardType[] => {
  const allWords = wordlist.split('\n').filter((word: string) => word.trim() !== '');
  const gameCards: CardType[] = [];

  // Randomly select 25 words
  const selectedWords = [];
  const tempWords = [...allWords];
  for (let i = 0; i < 25; i++) {
    const randomIndex = Math.floor(Math.random() * tempWords.length);
    selectedWords.push(tempWords[randomIndex]);
    tempWords.splice(randomIndex, 1);
  }

  // Team assignment counts for randomization
  const teams: CardColor[] = [
    ...Array(9).fill('red'),
    ...Array(8).fill('blue'),
    ...Array(1).fill('black'),
    ...Array(7).fill('neutral'),
  ];

  // Randomly assign teams to words
  selectedWords.forEach((word) => {
    const randomIndex = Math.floor(Math.random() * teams.length);
    gameCards.push({
      word,
      color: teams[randomIndex],
      isRevealed: false,
      wasRecentlyRevealed: false,
    });
    teams.splice(randomIndex, 1);
  });

  return gameCards;
};

// Select four random agents to form the two teams
const selectRandomAgents = (activeModels: LLMModel[] = []): GameAgents => {
  const availableAgents = [...activeModels];
  if (availableAgents.length < MIN_ACTIVE_MODELS) {
    throw new Error(
      `At least ${MIN_ACTIVE_MODELS} active models are required. Update your active model settings.`,
    );
  }

  const pickRandomAgent = () => {
    const randomIndex = Math.floor(Math.random() * availableAgents.length);
    return availableAgents.splice(randomIndex, 1)[0];
  };

  return {
    red: {
      spymaster: pickRandomAgent(),
      operative: pickRandomAgent(),
    },
    blue: {
      spymaster: pickRandomAgent(),
      operative: pickRandomAgent(),
    },
  } satisfies GameAgents;
};

const resetAnimations = (cards: CardType[]) => {
  cards.forEach((card) => {
    card.wasRecentlyRevealed = false;
  });
};

/** Trim and strip outer punctuation so model output still matches board words. */
function normalizeGuessToken(value: string): string {
  return value
    .trim()
    .normalize('NFKC')
    .replace(/^[\s"'`.,;:!?]+|[\s"'`.,;:!?]+$/g, '');
}

function findCardForGuess(cards: CardType[], guess: string): CardType | undefined {
  const g = normalizeGuessToken(guess).toUpperCase();
  if (!g) {
    return undefined;
  }

  return cards.find((c) => normalizeGuessToken(c.word).toUpperCase() === g);
}

// Set the guess properties and switch to operative role
export function updateGameStateFromSpymasterMove(
  currentState: GameState,
  move: SpymasterMove,
): GameState {
  const newState = { ...currentState };
  newState.currentClue = {
    clueText: move.clue.toUpperCase(),
    number: move.number,
  };
  newState.chatHistory.push({
    message: formatMoveMessage('spymaster', move),
    model: currentState.agents[currentState.currentTeam].spymaster,
    team: currentState.currentTeam,
    cards: currentState.cards,
  });
  newState.currentRole = 'operative';
  newState.currentGuesses = undefined;
  newState.previousRole = currentState.currentRole;
  newState.previousTeam = currentState.currentTeam;
  return newState;
}

function appendOperativeChatMessage(
  newState: GameState,
  snapshotState: GameState,
  move: OperativeMove,
  appliedGuesses: string[],
) {
  newState.currentGuesses = appliedGuesses;
  newState.chatHistory.push({
    message: formatMoveMessage('operative', { ...move, guesses: appliedGuesses }),
    model: snapshotState.agents[snapshotState.currentTeam].operative,
    team: snapshotState.currentTeam,
    cards: newState.cards,
  });
}

// Make guesses and switch to spymaster role
export function updateGameStateFromOperativeMove(
  currentState: GameState,
  move: OperativeMove,
): GameState {
  const newState = { ...currentState };

  // Reset recently revealed cards
  resetAnimations(newState.cards);

  const appliedGuesses: string[] = [];

  for (const guess of move.guesses) {
    const card = findCardForGuess(newState.cards, guess);

    // If card not found or already revealed, it's an invalid guess
    if (!card || card.isRevealed) {
      console.error(`INVALID GUESS: ${guess}`);
      continue;
    }

    appliedGuesses.push(card.word);

    card.isRevealed = true;
    card.wasRecentlyRevealed = true;

    newState.previousRole = currentState.currentRole;
    newState.previousTeam = currentState.currentTeam;

    // Assassin card instantly loses the game
    if (card.color === 'black') {
      newState.gameWinner = currentState.currentTeam === 'red' ? 'blue' : 'red';
      resetAnimations(newState.cards);
      appendOperativeChatMessage(newState, currentState, move, appliedGuesses);
      return newState;
    }

    // Decrement the count of remaining cards for the team
    if (card.color === 'red') {
      newState.remainingRed--;
    } else if (card.color === 'blue') {
      newState.remainingBlue--;
    }

    // If no more cards remain for the team, they win
    if (newState.remainingRed === 0) {
      newState.gameWinner = 'red';
      appendOperativeChatMessage(newState, currentState, move, appliedGuesses);
      return newState;
    } else if (newState.remainingBlue === 0) {
      newState.gameWinner = 'blue';
      resetAnimations(newState.cards);
      appendOperativeChatMessage(newState, currentState, move, appliedGuesses);
      return newState;
    }

    // If we guessed a card that isn't our team's color, we're done
    if (card.color !== currentState.currentTeam) {
      break;
    }
  }

  // Switch to the other team's spymaster once we're done guessing
  newState.currentRole = 'spymaster';
  newState.currentTeam = currentState.currentTeam === 'red' ? 'blue' : 'red';
  // newState.currentClue = undefined;

  appendOperativeChatMessage(newState, currentState, move, appliedGuesses);
  return newState;
}
