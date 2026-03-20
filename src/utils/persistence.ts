import { ChatMessage } from './game';
import { AssistantPrefill } from './llm';
import { CardType, GameState, Role, TeamColor, initializeGameState } from './game';
import { getActiveModels, LLMModel, resolveModel } from './models';

type PersistedViewState =
  | 'game_start'
  | 'ready_for_turn'
  | 'waiting_for_response'
  | 'error'
  | 'game_over';

type PersistedModelRef = {
  id: string;
};

type PersistedChatMessage = Omit<ChatMessage, 'model'> & {
  model: PersistedModelRef;
};

type PersistedTeamAgents = {
  [key in Role]: PersistedModelRef;
};

type PersistedGameState = Omit<GameState, 'agents' | 'chatHistory'> & {
  agents: {
    [key in TeamColor]: PersistedTeamAgents;
  };
  chatHistory: PersistedChatMessage[];
};

export type PersistedPendingChatMessage = {
  turnKey: string;
  team: TeamColor;
  role: Role;
  reasoning: string;
  turnId?: string;
  prefill: AssistantPrefill;
};

export type SavedGameRecord = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  gameState: GameState;
  appState: Exclude<PersistedViewState, 'waiting_for_response'>;
  isGamePaused: boolean;
  errorMessage: string | null;
  pendingChatMessage: PersistedPendingChatMessage | null;
};

type PersistedSavedGameRecord = Omit<SavedGameRecord, 'gameState'> & {
  appState: PersistedViewState;
  gameState: PersistedGameState;
};

type PersistedGameLibrary = {
  version: 2;
  activeGameId: string;
  games: PersistedSavedGameRecord[];
};

type LegacyPersistedAppState = {
  version: 1;
  savedAt: string;
  gameState: PersistedGameState;
  appState: PersistedViewState;
  isGamePaused: boolean;
  errorMessage: string | null;
  pendingChatMessage: PersistedPendingChatMessage | null;
};

export type RestoredGameLibrary = {
  activeGameId: string;
  games: SavedGameRecord[];
};

const STORAGE_KEY = 'llm-codenames:app-state:v1';

export function loadPersistedGameLibrary(): RestoredGameLibrary | null {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    const rawValue = window.localStorage.getItem(STORAGE_KEY);
    if (!rawValue) {
      return null;
    }

    const parsed = JSON.parse(rawValue) as
      | Partial<PersistedGameLibrary>
      | Partial<LegacyPersistedAppState>;

    if (parsed.version === 2) {
      return deserializeGameLibrary(parsed);
    }

    if (parsed.version === 1) {
      return migrateLegacyState(parsed);
    }

    clearPersistedGameLibrary();
    return null;
  } catch {
    clearPersistedGameLibrary();
    return null;
  }
}

export function persistGameLibrary(payload: RestoredGameLibrary) {
  if (typeof window === 'undefined') {
    return;
  }

  const serializedLibrary: PersistedGameLibrary = {
    version: 2,
    activeGameId: payload.activeGameId,
    games: payload.games.map((game) => ({
      ...game,
      gameState: serializeGameState(game.gameState),
    })),
  };

  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(serializedLibrary));
}

export function clearPersistedGameLibrary() {
  if (typeof window === 'undefined') {
    return;
  }

  window.localStorage.removeItem(STORAGE_KEY);
}

export function createSavedGame(options: {
  title?: string;
  createdAt?: string;
  updatedAt?: string;
  gameState?: GameState;
  appState?: SavedGameRecord['appState'];
  isGamePaused?: boolean;
  errorMessage?: string | null;
  pendingChatMessage?: PersistedPendingChatMessage | null;
} = {}): SavedGameRecord {
  const timestamp = new Date().toISOString();

  return {
    id: createSavedGameId(),
    title: options.title ?? 'Game',
    createdAt: options.createdAt ?? timestamp,
    updatedAt: options.updatedAt ?? timestamp,
    gameState: options.gameState ?? initializeGameState(getActiveModels()),
    appState: options.appState ?? 'game_start',
    isGamePaused: options.isGamePaused ?? true,
    errorMessage: options.errorMessage ?? null,
    pendingChatMessage: options.pendingChatMessage ?? null,
  };
}

export function createNextGameTitle(games: SavedGameRecord[]) {
  const gameNumbers = games
    .map((game) => /^Game (\d+)$/i.exec(game.title)?.[1])
    .map((value) => Number.parseInt(value ?? '', 10))
    .filter((value) => Number.isInteger(value));

  const nextNumber = (gameNumbers.length ? Math.max(...gameNumbers) : 0) + 1;
  return `Game ${nextNumber}`;
}

function deserializeGameLibrary(rawLibrary: Partial<PersistedGameLibrary>): RestoredGameLibrary | null {
  if (!rawLibrary.activeGameId || !Array.isArray(rawLibrary.games)) {
    clearPersistedGameLibrary();
    return null;
  }

  const games = rawLibrary.games
    .map((game) => deserializeSavedGameRecord(game))
    .filter((game): game is SavedGameRecord => Boolean(game));

  if (!games.length) {
    clearPersistedGameLibrary();
    return null;
  }

  const activeGameId =
    games.some((game) => game.id === rawLibrary.activeGameId) ? rawLibrary.activeGameId : games[0].id;

  return {
    activeGameId,
    games,
  };
}

function migrateLegacyState(rawLegacyState: Partial<LegacyPersistedAppState>): RestoredGameLibrary | null {
  if (!rawLegacyState.gameState) {
    clearPersistedGameLibrary();
    return null;
  }

  const gameState = deserializeGameState(rawLegacyState.gameState);
  if (!gameState) {
    clearPersistedGameLibrary();
    return null;
  }

  const restoredGame = normalizeSavedGameRecord(
    createSavedGame({
      title: 'Recovered Game',
      createdAt:
        typeof rawLegacyState.savedAt === 'string' ? rawLegacyState.savedAt : new Date().toISOString(),
      updatedAt:
        typeof rawLegacyState.savedAt === 'string' ? rawLegacyState.savedAt : new Date().toISOString(),
      gameState,
      appState: isPersistedViewState(rawLegacyState.appState) ?
          normalizeViewState(rawLegacyState.appState, gameState)
        : 'ready_for_turn',
      isGamePaused: Boolean(rawLegacyState.isGamePaused),
      errorMessage: typeof rawLegacyState.errorMessage === 'string' ? rawLegacyState.errorMessage : null,
      pendingChatMessage: deserializePendingChatMessage(rawLegacyState.pendingChatMessage),
    }),
  );

  return {
    activeGameId: restoredGame.id,
    games: [restoredGame],
  };
}

function serializeGameState(gameState: GameState): PersistedGameState {
  return {
    ...gameState,
    agents: {
      red: {
        spymaster: { id: gameState.agents.red.spymaster.id },
        operative: { id: gameState.agents.red.operative.id },
      },
      blue: {
        spymaster: { id: gameState.agents.blue.spymaster.id },
        operative: { id: gameState.agents.blue.operative.id },
      },
    },
    chatHistory: gameState.chatHistory.map((message) => ({
      ...message,
      model: { id: message.model.id },
    })),
  };
}

function deserializeSavedGameRecord(
  rawRecord: Partial<PersistedSavedGameRecord>,
): SavedGameRecord | null {
  if (
    typeof rawRecord.id !== 'string' ||
    typeof rawRecord.title !== 'string' ||
    typeof rawRecord.createdAt !== 'string' ||
    typeof rawRecord.updatedAt !== 'string' ||
    !rawRecord.gameState
  ) {
    return null;
  }

  const gameState = deserializeGameState(rawRecord.gameState);
  if (!gameState) {
    return null;
  }

  const appState =
    isPersistedViewState(rawRecord.appState) ?
      normalizeViewState(rawRecord.appState, gameState)
    : 'ready_for_turn';

  const record: SavedGameRecord = {
    id: rawRecord.id,
    title: rawRecord.title,
    createdAt: rawRecord.createdAt,
    updatedAt: rawRecord.updatedAt,
    gameState,
    appState,
    isGamePaused: gameState.gameWinner ? true : appState === 'error' ? true : Boolean(rawRecord.isGamePaused),
    errorMessage: typeof rawRecord.errorMessage === 'string' ? rawRecord.errorMessage : null,
    pendingChatMessage: deserializePendingChatMessage(rawRecord.pendingChatMessage),
  };

  return normalizeSavedGameRecord(record);
}

function normalizeSavedGameRecord(record: SavedGameRecord): SavedGameRecord {
  return {
    ...record,
    appState: normalizeViewState(record.appState, record.gameState),
    isGamePaused:
      record.gameState.gameWinner ? true
      : record.appState === 'error' ? true
      : record.isGamePaused,
    pendingChatMessage:
      record.pendingChatMessage?.turnKey ? record.pendingChatMessage : null,
  };
}

function normalizeViewState(
  appState: PersistedViewState,
  gameState: GameState,
): SavedGameRecord['appState'] {
  if (gameState.gameWinner) {
    return 'game_over';
  }

  if (appState === 'waiting_for_response') {
    return 'ready_for_turn';
  }

  return appState;
}

function deserializeGameState(rawState: PersistedGameState): GameState | null {
  const redSpymaster = resolvePersistedModel(rawState.agents?.red?.spymaster?.id);
  const redOperative = resolvePersistedModel(rawState.agents?.red?.operative?.id);
  const blueSpymaster = resolvePersistedModel(rawState.agents?.blue?.spymaster?.id);
  const blueOperative = resolvePersistedModel(rawState.agents?.blue?.operative?.id);

  if (!redSpymaster || !redOperative || !blueSpymaster || !blueOperative) {
    return null;
  }

  if (!Array.isArray(rawState.cards) || !Array.isArray(rawState.chatHistory)) {
    return null;
  }

  return {
    ...rawState,
    agents: {
      red: {
        spymaster: redSpymaster,
        operative: redOperative,
      },
      blue: {
        spymaster: blueSpymaster,
        operative: blueOperative,
      },
    },
    cards: rawState.cards.map((card) => sanitizeCard(card)),
    chatHistory: rawState.chatHistory
      .map((message) => deserializeChatMessage(message))
      .filter((message): message is ChatMessage => Boolean(message)),
  };
}

function deserializeChatMessage(rawMessage: PersistedChatMessage): ChatMessage | null {
  const model = resolvePersistedModel(rawMessage.model?.id);
  if (!model || typeof rawMessage.message !== 'string') {
    return null;
  }

  return {
    ...rawMessage,
    model,
    cards: Array.isArray(rawMessage.cards) ? rawMessage.cards.map((card) => sanitizeCard(card)) : undefined,
  };
}

function deserializePendingChatMessage(
  rawPending: PersistedPendingChatMessage | null | undefined,
): PersistedPendingChatMessage | null {
  if (!rawPending || typeof rawPending !== 'object') {
    return null;
  }

  if (
    typeof rawPending.turnKey !== 'string' ||
    typeof rawPending.reasoning !== 'string' ||
    !isTeamColor(rawPending.team) ||
    !isRole(rawPending.role) ||
    !rawPending.prefill ||
    typeof rawPending.prefill !== 'object' ||
    typeof rawPending.prefill.content !== 'string'
  ) {
    return null;
  }

  return {
    turnKey: rawPending.turnKey,
    team: rawPending.team,
    role: rawPending.role,
    reasoning: rawPending.reasoning,
    ...(typeof rawPending.turnId === 'string' ?
      { turnId: rawPending.turnId }
      : {}),
    prefill: {
      content: rawPending.prefill.content,
      reasoning:
        typeof rawPending.prefill.reasoning === 'string' ? rawPending.prefill.reasoning : undefined,
      reasoningDetails:
        Array.isArray(rawPending.prefill.reasoningDetails) ?
          rawPending.prefill.reasoningDetails
        : undefined,
    },
  };
}

function sanitizeCard(card: CardType): CardType {
  return {
    word: card.word,
    color: card.color,
    isRevealed: Boolean(card.isRevealed),
    wasRecentlyRevealed: Boolean(card.wasRecentlyRevealed),
  };
}

function resolvePersistedModel(modelId: string | undefined): LLMModel | null {
  return resolveModel(modelId);
}

function createSavedGameId() {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto ?
      crypto.randomUUID()
    : `game-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function isPersistedViewState(value: unknown): value is PersistedViewState {
  return (
    value === 'game_start' ||
    value === 'ready_for_turn' ||
    value === 'waiting_for_response' ||
    value === 'error' ||
    value === 'game_over'
  );
}

function isTeamColor(value: unknown): value is TeamColor {
  return value === 'red' || value === 'blue';
}

function isRole(value: unknown): value is Role {
  return value === 'spymaster' || value === 'operative';
}
