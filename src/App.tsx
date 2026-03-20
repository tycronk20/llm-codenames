import {
  ChevronLeft,
  ChevronRight,
  FolderKanban,
  Loader2,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Settings,
  Square,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import Card from './components/Card';
import { Chat } from './components/Chat';
import { Scoreboard } from './components/Scoreboard';
import { SettingsModal } from './components/SettingsModal';
import {
  GameState,
  initializeGameState,
  OperativeMove,
  SpymasterMove,
  updateGameStateFromOperativeMove,
  updateGameStateFromSpymasterMove,
} from './utils/game';
import {
  AssistantPrefill,
  createLlmRequest,
  reconnectLLMResponse,
  streamLLMResponse,
} from './utils/llm';
import {
  createNextGameTitle,
  createSavedGame,
  loadPersistedGameLibrary,
  persistGameLibrary,
  type SavedGameRecord,
} from './utils/persistence';
import {
  getActiveModels,
} from './utils/models';

type AppState = 'game_start' | 'ready_for_turn' | 'waiting_for_response' | 'error' | 'game_over';
type PendingRequest = {
  modelName: string;
  team: GameState['currentTeam'];
  role: GameState['currentRole'];
  startedAt: number;
  resumeMode: 'fresh' | 'resume';
};
type PartialChatMessage = {
  turnKey: string;
  team: PendingRequest['team'];
  role: PendingRequest['role'];
  reasoning: string;
  turnId?: string;
  prefill: AssistantPrefill;
};

function createTurnKey(gameState: GameState) {
  return [
    gameState.chatHistory.length,
    gameState.currentTeam,
    gameState.currentRole,
    gameState.remainingRed,
    gameState.remainingBlue,
  ].join(':');
}

function createInitialLibrary() {
  const restoredLibrary = loadPersistedGameLibrary();
  if (restoredLibrary?.games.length) {
    return restoredLibrary;
  }

  const initialGame = createSavedGame({ title: 'Game 1' });
  return {
    activeGameId: initialGame.id,
    games: [initialGame],
  };
}

function formatSavedGameTimestamp(timestamp: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(timestamp));
}

function describeSavedGameState(savedGame: SavedGameRecord) {
  if (savedGame.gameState.gameWinner) {
    return `${savedGame.gameState.gameWinner.toUpperCase()} won`;
  }

  if (savedGame.appState === 'error') {
    return 'Error';
  }

  if (savedGame.isGamePaused) {
    return 'Paused';
  }

  return 'Live';
}

function createTurnId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }

  return `turn-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function App() {
  const [initialLibrary] = useState(createInitialLibrary);
  const initialActiveGame =
    initialLibrary.games.find((game) => game.id === initialLibrary.activeGameId) ?? initialLibrary.games[0];
  const [savedGames, setSavedGames] = useState<SavedGameRecord[]>(() => initialLibrary.games);
  const [activeGameId, setActiveGameId] = useState(() => initialActiveGame.id);
  const [gameState, setGameState] = useState<GameState>(
    () => initialActiveGame.gameState,
  );
  const [appState, setAppState] = useState<AppState>(
    () => initialActiveGame.appState,
  );
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isBrowserCollapsed, setIsBrowserCollapsed] = useState(false);
  const [isGamePaused, setIsGamePaused] = useState(
    () => initialActiveGame.isGamePaused,
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(
    () => initialActiveGame.errorMessage,
  );
  const [idleWarningMessage, setIdleWarningMessage] = useState<string | null>(null);
  const [pendingRequest, setPendingRequest] = useState<PendingRequest | null>(null);
  const [pendingChatMessage, setPendingChatMessage] = useState<PartialChatMessage | null>(
    () => initialActiveGame.pendingChatMessage,
  );
  const [streamedTokenCount, setStreamedTokenCount] = useState(0);
  const [streamConnected, setStreamConnected] = useState(false);
  const [requestEpoch, setRequestEpoch] = useState(0);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const activeTurnKeyRef = useRef<string | null>(null);
  const activeRequestControllerRef = useRef<AbortController | null>(null);
  const currentTurnKey = createTurnKey(gameState);
  const currentTurnPending =
    pendingChatMessage?.turnKey === currentTurnKey ? pendingChatMessage : null;
  const browserGames = [...savedGames].sort(
    (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
  );

  useEffect(() => {
    setPendingChatMessage((current) => current?.turnKey === currentTurnKey ? current : null);
  }, [currentTurnKey]);

  const cancelActiveRequest = () => {
    activeRequestControllerRef.current?.abort();
    activeRequestControllerRef.current = null;
    activeTurnKeyRef.current = null;
  };

  const resetTransientUi = () => {
    setIdleWarningMessage(null);
    setPendingRequest(null);
    setStreamedTokenCount(0);
    setStreamConnected(false);
  };

  const clearCurrentTurnPending = () => {
    setPendingChatMessage((current) => current?.turnKey === currentTurnKey ? null : current);
  };

  const retryCurrentTurn = () => {
    cancelActiveRequest();
    clearCurrentTurnPending();
    resetTransientUi();
    setErrorMessage(null);
    setRequestEpoch((value) => value + 1);
    setIsGamePaused(false);
    setAppState('ready_for_turn');
  };

  const hydrateSavedGame = (savedGame: SavedGameRecord, forcePaused = true) => {
    cancelActiveRequest();
    resetTransientUi();
    setActiveGameId(savedGame.id);
    setGameState(savedGame.gameState);
    setAppState(
      savedGame.gameState.gameWinner ? 'game_over'
        : savedGame.appState === 'error' ? 'error'
          : savedGame.appState,
    );
    setIsGamePaused(forcePaused ? true : savedGame.isGamePaused);
    setErrorMessage(savedGame.errorMessage);
    setPendingChatMessage(savedGame.pendingChatMessage);
  };

  const openSavedGame = (savedGame: SavedGameRecord) => {
    hydrateSavedGame(savedGame);
  };

  const startNewGame = () => {
    const newGame = createSavedGame({
      title: createNextGameTitle(savedGames),
    });

    setSavedGames((current) => [newGame, ...current]);
    hydrateSavedGame(newGame);
  };

  useEffect(() => {
    setSavedGames((current) => {
      let didUpdate = false;
      const nextGames = current.map((savedGame) => {
        if (savedGame.id !== activeGameId) {
          return savedGame;
        }

        didUpdate = true;
        return {
          ...savedGame,
          updatedAt: new Date().toISOString(),
          gameState,
          appState: appState === 'waiting_for_response' ? 'ready_for_turn' : appState,
          isGamePaused,
          errorMessage,
          pendingChatMessage,
        };
      });

      return didUpdate ? nextGames : current;
    });
  }, [activeGameId, appState, errorMessage, gameState, isGamePaused, pendingChatMessage]);

  useEffect(() => {
    persistGameLibrary({
      activeGameId,
      games: savedGames,
    });
  }, [activeGameId, savedGames]);

  useEffect(() => {
    if (isGamePaused) {
      cancelActiveRequest();
      setIdleWarningMessage(null);
      setPendingRequest(null);
      setStreamedTokenCount(0);
      setStreamConnected(false);
      if (appState === 'waiting_for_response') {
        setAppState('ready_for_turn');
      }
      return;
    }

    if (gameState.gameWinner) {
      cancelActiveRequest();
      setIdleWarningMessage(null);
      setPendingRequest(null);
      setPendingChatMessage(null);
      setStreamedTokenCount(0);
      setStreamConnected(false);
      setIsGamePaused(true);
      setAppState('game_over');
      return;
    }

    if (appState === 'ready_for_turn') {
      const turnKey = createTurnKey(gameState);
      const requestKey = `${turnKey}:${requestEpoch}`;

      if (activeTurnKeyRef.current === requestKey) {
        return;
      }

      const turnState = gameState;
      const activeModel = turnState.agents[turnState.currentTeam][turnState.currentRole];
      const controller = new AbortController();
      const savedPending =
        pendingChatMessage?.turnKey === turnKey ? pendingChatMessage : undefined;
      const turnId = savedPending?.turnId ?? createTurnId();
      const resumeMode = savedPending?.turnId ? 'resume' : 'fresh';
      activeTurnKeyRef.current = requestKey;
      activeRequestControllerRef.current = controller;
      setAppState('waiting_for_response');
      setPendingRequest({
        modelName: activeModel.modelName,
        team: turnState.currentTeam,
        role: turnState.currentRole,
        startedAt: Date.now(),
        resumeMode,
      });
      setPendingChatMessage((current) =>
        current?.turnKey === turnKey ?
          {
            ...current,
            turnId,
          }
        : null,
      );
      setErrorMessage(null);
      setIdleWarningMessage(null);
      setStreamedTokenCount(0);
      setStreamConnected(false);

      console.info('[llm] starting turn', {
        modelId: activeModel.id,
        team: turnState.currentTeam,
        role: turnState.currentRole,
        turnKey,
        resumeMode,
        turnId,
      });

      void (async () => {
        try {
          let completedMove: SpymasterMove | OperativeMove | null = null;
          const streamUpdates =
            resumeMode === 'resume' ?
              reconnectLLMResponse(turnId, controller.signal, {
                onIdleStateChange: (message) => {
                  if (activeRequestControllerRef.current !== controller || controller.signal.aborted) {
                    return;
                  }

                  setIdleWarningMessage(message);
                },
              })
            : streamLLMResponse(
                createLlmRequest(turnState, turnId),
                controller.signal,
                {
                  onIdleStateChange: (message) => {
                    if (activeRequestControllerRef.current !== controller || controller.signal.aborted) {
                      return;
                    }

                    setIdleWarningMessage(message);
                  },
                },
              );

          for await (const update of streamUpdates) {
            if (activeRequestControllerRef.current !== controller || controller.signal.aborted) {
              return;
            }

            if (update.type === 'stream_open') {
              setStreamConnected(true);
              setPendingChatMessage((current) =>
                current?.turnKey === turnKey ?
                  current
                  : {
                      turnKey,
                      team: turnState.currentTeam,
                      role: turnState.currentRole,
                      reasoning: '',
                      turnId,
                      prefill: { content: '' },
                    },
              );
              continue;
            }

            if (update.type === 'progress') {
              setStreamedTokenCount(update.tokenCount);
              continue;
            }

            if (update.type === 'prefill') {
              setPendingChatMessage((current) => ({
                turnKey,
                team: turnState.currentTeam,
                role: turnState.currentRole,
                reasoning: update.prefill.reasoning ?? current?.reasoning ?? '',
                turnId,
                prefill: update.prefill,
              }));
              continue;
            }

            if (update.type === 'reasoning') {
              setPendingChatMessage((current) => ({
                turnKey,
                team: turnState.currentTeam,
                role: turnState.currentRole,
                reasoning: update.reasoning,
                turnId,
                prefill: {
                  content: current?.prefill.content ?? '',
                  reasoning: update.reasoning,
                  reasoningDetails: current?.prefill.reasoningDetails,
                },
              }));
              continue;
            }

            if (update.type === 'complete') {
              completedMove = update.move;
            }
          }

          if (activeRequestControllerRef.current !== controller || controller.signal.aborted) {
            return;
          }
          if (!completedMove) {
            throw new Error('The LLM did not return a move.');
          }

          setPendingRequest(null);
          setPendingChatMessage(null);
          setIdleWarningMessage(null);
          setStreamedTokenCount(0);
          setStreamConnected(false);
          if (turnState.currentRole === 'spymaster') {
            setGameState(updateGameStateFromSpymasterMove(turnState, completedMove as SpymasterMove));
          } else {
            setGameState(updateGameStateFromOperativeMove(turnState, completedMove as OperativeMove));
          }
          setAppState('ready_for_turn');
        } catch (error) {
          if (controller.signal.aborted || activeRequestControllerRef.current !== controller) {
            return;
          }

          console.error('Error in fetchResponse:', error);
          setPendingRequest(null);
          setPendingChatMessage(null);
          setIdleWarningMessage(null);
          setStreamedTokenCount(0);
          setStreamConnected(false);
          setErrorMessage(
            error instanceof Error ?
              error.message
              : 'An unknown error occurred while calling the LLM.',
          );
          setAppState('error');
          setIsGamePaused(true);
        } finally {
          if (activeRequestControllerRef.current === controller) {
            activeRequestControllerRef.current = null;
            activeTurnKeyRef.current = null;
          }
        }
      })();

      return;
    }

    if (appState === 'game_start') {
      setAppState('ready_for_turn');
    }
  }, [appState, gameState, isGamePaused, pendingChatMessage, requestEpoch]);

  useEffect(() => {
    return () => {
      cancelActiveRequest();
    };
  }, []);

  // Handle scrolling to the bottom of the chat history as chats stream in
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [gameState, appState, pendingChatMessage, streamedTokenCount, streamConnected]);

  return (
    <div className='flex min-h-screen w-full flex-col bg-gradient-to-br from-slate-800 to-slate-600 antialiased lg:h-screen lg:flex-row lg:overflow-hidden'>
      <aside
        className={`flex flex-col border-b border-slate-500/30 bg-slate-950/40 backdrop-blur-sm transition-all duration-200 lg:h-screen lg:flex-none lg:border-b-0 lg:border-r ${isBrowserCollapsed ? 'w-full lg:w-[4.5rem]' : 'w-full lg:w-64'
          }`}
      >
        <div
          className={`shrink-0 border-slate-500/30 ${isBrowserCollapsed ? 'border-b px-3 py-3 lg:h-full lg:border-b-0' : 'border-b px-3 py-3'
            }`}
        >
          <div
            className={`flex gap-2 ${isBrowserCollapsed ? 'items-center justify-between lg:flex-col lg:items-stretch' : 'items-center justify-between'
              }`}
          >
            <div
              className={`flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.18em] text-slate-200 ${isBrowserCollapsed ? 'lg:justify-center' : ''
                }`}
            >
              <FolderKanban className='size-4' />
              {!isBrowserCollapsed && 'Games'}
            </div>
            <div
              className={`flex items-center gap-2 flex-wrap justify-end ${isBrowserCollapsed ? 'lg:flex-col lg:justify-start' : ''
                }`}
            >
              <button
                type='button'
                onClick={() => setIsBrowserCollapsed((current) => !current)}
                className='inline-flex items-center justify-center rounded-full border border-slate-500/30 bg-slate-800/50 p-1.5 text-slate-400 transition hover:bg-slate-700 hover:text-slate-200'
                title={isBrowserCollapsed ? 'Expand games browser' : 'Collapse games browser'}
                aria-label={isBrowserCollapsed ? 'Expand games browser' : 'Collapse games browser'}
              >
                {isBrowserCollapsed ?
                  <ChevronRight className='size-4' />
                  : <ChevronLeft className='size-4' />}
              </button>
              <button
                type='button'
                onClick={() => setIsSettingsOpen(true)}
                className='inline-flex items-center justify-center rounded-full border border-slate-500/30 bg-slate-800/50 p-1.5 text-slate-400 transition hover:bg-slate-700 hover:text-slate-200'
                title='Settings'
                aria-label='Settings'
              >
                <Settings className='size-4' />
              </button>
              <div className={`w-full flex ${isBrowserCollapsed ? 'justify-center' : 'justify-end lg:w-auto mt-2 lg:mt-0'}`}>
                <button
                  type='button'
                  onClick={startNewGame}
                  className={`inline-flex items-center justify-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-500/10 text-[11px] font-semibold uppercase tracking-[0.16em] text-emerald-100 transition hover:bg-emerald-500/20 ${isBrowserCollapsed ? 'p-2 w-full' : 'px-3 py-1.5 whitespace-nowrap'
                    }`}
                  title='New Game'
                  aria-label='New Game'
                >
                  <Plus className='size-3 shrink-0' />
                  {!isBrowserCollapsed && <span>New Game</span>}
                </button>
              </div>
            </div>
          </div>

        </div>

        {!isBrowserCollapsed && (
          <div className='flex-1 min-h-0 overflow-y-auto px-3 py-3 max-h-[18rem] lg:max-h-none'>
            <div className='space-y-2'>
              {browserGames.map((savedGame) => {
                const revealedRed = 9 - savedGame.gameState.remainingRed;
                const revealedBlue = 8 - savedGame.gameState.remainingBlue;

                return (
                  <button
                    key={savedGame.id}
                    type='button'
                    onClick={() => openSavedGame(savedGame)}
                    className={`flex w-full flex-col gap-1.5 rounded-xl border px-3 py-2.5 text-left transition ${savedGame.id === activeGameId ?
                      'border-sky-300/60 bg-sky-500/15 shadow-lg shadow-sky-950/20'
                      : 'border-slate-500/30 bg-slate-900/60 hover:border-slate-400/50 hover:bg-slate-900/80'
                      }`}
                  >
                    <div className='flex items-start justify-between gap-2'>
                      <div className='min-w-0'>
                        <div className='truncate text-sm font-semibold text-slate-100'>{savedGame.title}</div>
                        <div className='text-[10px] uppercase tracking-[0.16em] text-slate-400'>
                          {formatSavedGameTimestamp(savedGame.updatedAt)}
                        </div>
                      </div>
                      <span
                        className={`shrink-0 rounded-full px-2 py-1 text-[9px] font-semibold uppercase tracking-[0.16em] ${savedGame.gameState.gameWinner ?
                          'bg-emerald-500/15 text-emerald-200'
                          : savedGame.appState === 'error' ?
                            'bg-rose-500/15 text-rose-200'
                            : savedGame.isGamePaused ?
                              'bg-amber-500/15 text-amber-100'
                              : 'bg-sky-500/15 text-sky-100'
                          }`}
                      >
                        {describeSavedGameState(savedGame)}
                      </span>
                    </div>
                    <div className='truncate text-[11px] text-slate-300'>
                      {savedGame.gameState.currentTeam} {savedGame.gameState.currentRole}
                      {savedGame.gameState.currentClue &&
                        ` · ${savedGame.gameState.currentClue.clueText}, ${savedGame.gameState.currentClue.number}`}
                    </div>
                    <div className='flex items-center gap-2 text-[10px] uppercase tracking-[0.16em] text-slate-500'>
                      <span>Chat {savedGame.gameState.chatHistory.length}</span>
                      <span>R {revealedRed}</span>
                      <span>B {revealedBlue}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </aside>

      {/* Left panel: Scoreboard + Game Board + Game Controls */}
      <div className='flex w-full flex-col items-center gap-y-6 px-2 sm:mt-4 lg:min-w-0 lg:flex-1 lg:overflow-y-auto lg:gap-y-8 lg:px-4'>
        {appState === 'error' && (
          <div className='fixed left-4 top-4 z-50 rounded-md bg-red-500 px-4 py-2 text-white shadow-lg'>
            {errorMessage ?? 'An error occurred. Please reload the game.'}
          </div>
        )}
        {idleWarningMessage && appState !== 'error' && (
          <div className='fixed left-4 top-4 z-50 rounded-md bg-amber-500 px-4 py-2 text-slate-950 shadow-lg'>
            {idleWarningMessage}
          </div>
        )}

        {/* Scoreboard */}
        <Scoreboard gameState={gameState} />

        {/* Game board */}
        <div className='flex flex-col items-center justify-center'>
          <div className='grid w-full max-w-3xl grid-cols-5 gap-0.5 p-2 sm:gap-2 md:gap-4'>
            {gameState.cards.map((card, index) => (
              <Card
                key={index}
                word={card.word}
                color={card.color}
                isRevealed={card.isRevealed}
                wasRecentlyRevealed={card.wasRecentlyRevealed}
                isSpymasterView={true}
              />
            ))}
          </div>
        </div>

        {/* Start/Pause game button */}
        <button
          onClick={() => {
            const shouldResetGame = appState === 'game_over' || appState === 'error';
            const nextPausedState = shouldResetGame ? false : !isGamePaused;

            if (shouldResetGame || nextPausedState) {
              cancelActiveRequest();
              resetTransientUi();
            }
            if (appState === 'game_over' || appState === 'error') {
              setGameState(
                initializeGameState(getActiveModels()),
              );
              setAppState('game_start');
              setErrorMessage(null);
              setPendingChatMessage(null);
              resetTransientUi();
            }
            if (!shouldResetGame && !nextPausedState) {
              setRequestEpoch((value) => value + 1);
              setAppState('ready_for_turn');
            }
            setIsGamePaused(nextPausedState);
          }}
          className='mb-6 flex w-36 items-center justify-center gap-2 rounded bg-slate-200 px-2 py-2 font-bold text-slate-800 hover:bg-slate-300'
        >
          {appState === 'game_start' ?
            <>
              <Play className='inline size-4' /> Start
            </>
            : appState === 'game_over' || appState === 'error' ?
              <>
                <RotateCcw className='inline size-4' /> Restart
              </>
              : isGamePaused ?
                <>
                  <Play className='inline size-4' /> Continue
                </>
                : <>
                  <Pause className='inline size-4' /> Pause
                </>
          }
        </button>
      </div>

      {/* Right panel: Chat history */}
      <div
        ref={chatContainerRef}
        className='relative w-full bg-slate-800/50 p-2 backdrop-blur-sm md:max-h-[32rem] md:overflow-y-auto lg:h-screen lg:w-[22rem] lg:border-l lg:border-slate-500/30'
      >
        {/* Spinner & Pause indicator */}
        {appState === 'waiting_for_response' && (
          <div className='sticky top-0 z-10 mb-2 -mx-2 -mt-2 flex w-full items-center justify-between border-b border-slate-500/40 bg-slate-900/95 px-4 py-3 shadow-sm backdrop-blur-md'>
            <div className='flex min-w-0 flex-1 flex-col gap-0.5'>
              <div className='flex items-center gap-2'>
                <Loader2 className='size-4 shrink-0 animate-spin text-slate-200' />
                <span className='min-w-0 truncate text-sm text-slate-200'>
                  <span className='font-medium text-slate-100'>
                    {pendingRequest?.modelName ?? 'LLM'}
                  </span>
                  <span className='text-slate-400'> · reasoning...</span>
                </span>
              </div>
              <span className='ml-6 text-[10px] uppercase tracking-[0.16em] text-slate-500'>
                ~{streamedTokenCount} toks streamed
              </span>
            </div>
            <div className='flex gap-2'>
              <button
                type='button'
                onClick={() => {
                  cancelActiveRequest();
                  setIdleWarningMessage(null);
                  setPendingRequest(null);
                  setStreamedTokenCount(0);
                  setStreamConnected(false);
                  setIsGamePaused(true);
                  setAppState('ready_for_turn');
                }}
                className='flex items-center gap-1 rounded bg-amber-500/20 px-2 py-1 text-[11px] font-bold uppercase tracking-[0.16em] text-amber-200 transition hover:bg-amber-500/30'
              >
                <Square className='size-3 fill-current' />
                Pause
              </button>
              <button
                type='button'
                onClick={retryCurrentTurn}
                className='flex items-center gap-1 rounded border border-rose-400/40 bg-rose-500/10 px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-rose-100 transition hover:bg-rose-500/20'
              >
                <RotateCcw className='size-3' />
                Retry
              </button>
            </div>
          </div>
        )}

        {isGamePaused && appState === 'ready_for_turn' && (
          <div className='sticky top-0 z-10 mb-2 -mx-2 -mt-2 flex w-full flex-col border-b border-amber-500/40 bg-slate-900/95 px-4 py-3 text-xs font-semibold tracking-wide text-slate-100 shadow-sm backdrop-blur-md'>
            <div className='flex items-center justify-between'>
              <div className='flex flex-col'>
                <div className='flex items-center gap-2 text-amber-500'>
                  <Pause className='size-4 fill-current' />
                  <span>{currentTurnPending ? 'Paused with resumable stream' : 'Paused'}</span>
                </div>
                {currentTurnPending?.turnId && (
                  <span className='ml-6 mt-1 text-[10px] uppercase tracking-[0.16em] text-slate-500'>
                    turn stream available
                  </span>
                )}
              </div>
              <div className='flex items-center gap-2'>
                <button
                  type='button'
                  onClick={() => {
                    setRequestEpoch((value) => value + 1);
                    setAppState('ready_for_turn');
                    setIsGamePaused(false);
                  }}
                  className='flex items-center gap-1 rounded bg-amber-500/20 px-3 py-1.5 text-[11px] font-bold uppercase tracking-[0.16em] text-amber-200 transition hover:bg-amber-500/30'
                >
                  <Play className='size-3 fill-current' />
                  Resume
                </button>
                {currentTurnPending && (
                  <button
                    type='button'
                    onClick={retryCurrentTurn}
                    className='flex items-center gap-1 rounded border border-rose-400/40 bg-rose-500/10 px-2 py-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-rose-100 transition hover:bg-rose-500/20'
                  >
                    <RotateCcw className='size-3' />
                    Retry Fresh
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
        {gameState.chatHistory.map((message, index) => (
          <Chat key={index} {...message} />
        ))}
        {pendingRequest &&
          pendingChatMessage &&
          (streamConnected || pendingChatMessage.reasoning.trim()) && (
            <Chat
              message={pendingChatMessage.reasoning}
              model={gameState.agents[pendingChatMessage.team][pendingChatMessage.role]}
              team={pendingChatMessage.team}
              cards={gameState.cards}
              isStreaming={Boolean(pendingRequest)}
            />
          )}
        {appState === 'game_over' && (
          <div className='flex w-full justify-center p-2 font-semibold tracking-wide'>
            <div
              className={`text-${gameState.gameWinner === 'red' ? 'rose' : 'sky'}-500 text-base`}
            >
              {gameState.gameWinner === 'red' ? 'Red' : 'Blue'} team wins.
            </div>
          </div>
        )}

      </div>

      <SettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
    </div>
  );
}
