import { FolderKanban, Loader2, Pause, Play, Plus, RotateCcw, Settings, Square } from 'lucide-react';
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
import { AssistantPrefill, createMessagesFromGameState, streamLLMResponse } from './utils/llm';
import {
  createNextGameTitle,
  createSavedGame,
  loadPersistedGameLibrary,
  persistGameLibrary,
  type SavedGameRecord,
} from './utils/persistence';

type AppState = 'game_start' | 'ready_for_turn' | 'waiting_for_response' | 'error' | 'game_over';
type PendingRequest = {
  modelName: string;
  team: GameState['currentTeam'];
  role: GameState['currentRole'];
  startedAt: number;
  resumeMode: 'fresh' | 'prefill' | 'prefill-disabled';
};
type PartialChatMessage = {
  turnKey: string;
  team: PendingRequest['team'];
  role: PendingRequest['role'];
  reasoning: string;
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
  const [hiddenThinkingUpdateCount, setHiddenThinkingUpdateCount] = useState(0);
  const [requestAgeSeconds, setRequestAgeSeconds] = useState(0);
  const [requestEpoch, setRequestEpoch] = useState(0);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const activeTurnKeyRef = useRef<string | null>(null);
  const activeRequestControllerRef = useRef<AbortController | null>(null);
  const pendingChatMessageRef = useRef<PartialChatMessage | null>(null);
  const lastIdleResumeSignatureRef = useRef<string | null>(null);
  const currentTurnKey = createTurnKey(gameState);
  const currentTurnPrefill =
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
    setHiddenThinkingUpdateCount(0);
    lastIdleResumeSignatureRef.current = null;
  };

  const clearCurrentTurnPrefill = () => {
    setPendingChatMessage((current) => current?.turnKey === currentTurnKey ? null : current);
    lastIdleResumeSignatureRef.current = null;
  };

  const retryCurrentTurn = () => {
    cancelActiveRequest();
    clearCurrentTurnPrefill();
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
    setRequestAgeSeconds(0);
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
    pendingChatMessageRef.current = pendingChatMessage;
  }, [pendingChatMessage]);

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
      setHiddenThinkingUpdateCount(0);
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
      setHiddenThinkingUpdateCount(0);
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
      const savedPrefill =
        pendingChatMessage?.turnKey === turnKey ? pendingChatMessage.prefill : undefined;
      const assistantPrefill =
        savedPrefill && activeModel.openRouterAssistantPrefillEnabled !== false ?
          savedPrefill
        : undefined;
      const resumeMode =
        assistantPrefill ? 'prefill'
        : savedPrefill ? 'prefill-disabled'
        : 'fresh';
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
      setPendingChatMessage((current) => current?.turnKey === turnKey ? current : null);
      setErrorMessage(null);
      setIdleWarningMessage(null);
      setStreamedTokenCount(0);
      setHiddenThinkingUpdateCount(savedPrefill?.reasoningDetails?.length ?? 0);

      console.info('[llm] starting turn', {
        modelId: activeModel.id,
        team: turnState.currentTeam,
        role: turnState.currentRole,
        turnKey,
        resumeMode,
        savedPrefillContentChars: savedPrefill?.content.length ?? 0,
        savedReasoningDetails: savedPrefill?.reasoningDetails?.length ?? 0,
      });

      void (async () => {
        try {
          let completedMove: SpymasterMove | OperativeMove | null = null;
          for await (const update of streamLLMResponse(
            {
              messages: createMessagesFromGameState(turnState, assistantPrefill),
              modelId: activeModel.id,
              role: turnState.currentRole,
            },
            controller.signal,
            {
              onIdleStateChange: (message) => {
                if (activeRequestControllerRef.current !== controller || controller.signal.aborted) {
                  return;
                }

                if (!message) {
                  setIdleWarningMessage(null);
                  return;
                }

                const latestPending = pendingChatMessageRef.current;
                const savedPrefillForTurn =
                  latestPending?.turnKey === turnKey ? latestPending.prefill : undefined;
                const prefillSignature =
                  savedPrefillForTurn ?
                    `${turnKey}:${savedPrefillForTurn.content.length}:${savedPrefillForTurn.reasoning?.length ?? 0}:${savedPrefillForTurn.reasoningDetails?.length ?? 0}`
                  : null;

                if (
                  activeModel.autoResumeOnIdle !== false &&
                  activeModel.openRouterAssistantPrefillEnabled !== false &&
                  savedPrefillForTurn &&
                  (savedPrefillForTurn.content.trim() || savedPrefillForTurn.reasoningDetails?.length) &&
                  prefillSignature &&
                  lastIdleResumeSignatureRef.current !== prefillSignature
                ) {
                  lastIdleResumeSignatureRef.current = prefillSignature;
                  console.info('[llm] idle detected, auto-resuming from saved prefill', {
                    modelId: activeModel.id,
                    turnKey,
                    prefillContentChars: savedPrefillForTurn.content.length,
                    reasoningDetails: savedPrefillForTurn.reasoningDetails?.length ?? 0,
                  });
                  cancelActiveRequest();
                  setIdleWarningMessage('No output for 15s. Resuming from saved prefill...');
                  setPendingRequest(null);
                  setStreamedTokenCount(0);
                  setHiddenThinkingUpdateCount(0);
                  setRequestEpoch((value) => value + 1);
                  setAppState('ready_for_turn');
                  setIsGamePaused(false);
                  return;
                }

                setIdleWarningMessage(message);
              },
            },
          )) {
            if (activeRequestControllerRef.current !== controller || controller.signal.aborted) {
              return;
            }

            if (update.type === 'progress') {
              setStreamedTokenCount(update.tokenCount);
              continue;
            }

            if (update.type === 'prefill') {
              setHiddenThinkingUpdateCount(update.prefill.reasoningDetails?.length ?? 0);
              setPendingChatMessage((current) => ({
                turnKey,
                team: turnState.currentTeam,
                role: turnState.currentRole,
                reasoning: update.prefill.reasoning ?? current?.reasoning ?? '',
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
                prefill: {
                  content: current?.prefill.content ?? '',
                  reasoning: update.reasoning,
                  reasoningDetails: current?.prefill.reasoningDetails,
                },
              }));
              continue;
            }

            completedMove = update.move;
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
          setHiddenThinkingUpdateCount(0);
          lastIdleResumeSignatureRef.current = null;
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
          setHiddenThinkingUpdateCount(0);
          lastIdleResumeSignatureRef.current = null;
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
  }, [appState, gameState, isGamePaused, requestEpoch]);

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
  }, [gameState, appState]);

  useEffect(() => {
    if (!pendingRequest) {
      setRequestAgeSeconds(0);
      return;
    }

    setRequestAgeSeconds(Math.floor((Date.now() - pendingRequest.startedAt) / 1000));
    const interval = window.setInterval(() => {
      setRequestAgeSeconds(Math.floor((Date.now() - pendingRequest.startedAt) / 1000));
    }, 1000);

    return () => window.clearInterval(interval);
  }, [pendingRequest]);

  return (
    <div className='flex min-h-screen flex-col bg-gradient-to-br from-slate-800 to-slate-600 antialiased lg:flex-row'>
      {/* Left panel: Scoreboard + Game Board + Game Controls */}
      <div className='flex w-full flex-col items-center gap-y-6 px-2 sm:mt-4 lg:min-w-0 lg:flex-1 lg:gap-y-8 lg:px-4'>
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
              setGameState(initializeGameState());
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

      <aside className='w-full border-t border-slate-500/30 bg-slate-950/35 backdrop-blur-sm lg:w-80 lg:border-l lg:border-t-0'>
        <div className='flex items-center justify-between border-b border-slate-500/30 px-4 py-4'>
          <div className='flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.18em] text-slate-200'>
            <FolderKanban className='size-4' />
            Games
          </div>
          <div className='flex items-center gap-2'>
            <button
              type='button'
              onClick={() => setIsSettingsOpen(true)}
              className='inline-flex items-center justify-center rounded-full border border-slate-500/30 bg-slate-800/50 p-1.5 text-slate-400 transition hover:bg-slate-700 hover:text-slate-200'
              title='Settings'
            >
              <Settings className='size-4' />
            </button>
            <button
              type='button'
              onClick={startNewGame}
              className='inline-flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-500/10 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-emerald-100 transition hover:bg-emerald-500/20'
            >
              <Plus className='size-3' />
              New Game
            </button>
          </div>
        </div>
        <div className='max-h-[18rem] overflow-y-auto p-3 lg:max-h-[calc(100vh-4.5rem)]'>
          <div className='space-y-2'>
            {browserGames.map((savedGame) => {
              const revealedRed = 9 - savedGame.gameState.remainingRed;
              const revealedBlue = 8 - savedGame.gameState.remainingBlue;

              return (
                <button
                  key={savedGame.id}
                  type='button'
                  onClick={() => openSavedGame(savedGame)}
                  className={`flex w-full flex-col gap-2 rounded-2xl border px-4 py-3 text-left transition ${
                    savedGame.id === activeGameId ?
                      'border-sky-300/60 bg-sky-500/15 shadow-lg shadow-sky-950/20'
                    : 'border-slate-500/30 bg-slate-900/60 hover:border-slate-400/50 hover:bg-slate-900/80'
                  }`}
                >
                  <div className='flex items-start justify-between gap-3'>
                    <div>
                      <div className='text-sm font-semibold text-slate-100'>{savedGame.title}</div>
                      <div className='text-[11px] uppercase tracking-[0.16em] text-slate-400'>
                        {formatSavedGameTimestamp(savedGame.updatedAt)}
                      </div>
                    </div>
                    <span
                      className={`rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] ${
                        savedGame.gameState.gameWinner ?
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
                  <div className='text-xs text-slate-300'>
                    {savedGame.gameState.currentTeam} {savedGame.gameState.currentRole}
                    {savedGame.gameState.currentClue &&
                      ` · ${savedGame.gameState.currentClue.clueText}, ${savedGame.gameState.currentClue.number}`}
                  </div>
                  <div className='flex items-center gap-3 text-[11px] uppercase tracking-[0.16em] text-slate-500'>
                    <span>Chat {savedGame.gameState.chatHistory.length}</span>
                    <span>Red {revealedRed}</span>
                    <span>Blue {revealedBlue}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </aside>

      {/* Right panel: Chat history */}
      <div
        ref={chatContainerRef}
        className='relative w-full bg-slate-800/50 p-2 backdrop-blur-sm md:max-h-[32rem] md:overflow-y-auto lg:h-screen lg:w-[24rem] lg:border-l lg:border-slate-500/30'
      >
        {gameState.chatHistory.map((message, index) => (
          <Chat key={index} {...message} />
        ))}
        {pendingChatMessage?.reasoning.trim() && (
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
        {/* Spinner & Pause indicator */}
        {appState === 'waiting_for_response' && (
          <div className='sticky flex w-full justify-end p-2'>
            <div className='flex items-center gap-3 rounded-full border border-slate-500/40 bg-slate-900/90 px-4 py-2 text-xs font-semibold tracking-wide text-slate-100 shadow-lg'>
              <div className='flex flex-col text-right'>
                <span>{pendingRequest?.modelName ?? 'LLM'} thinking</span>
                <span className='text-[11px] font-medium uppercase tracking-[0.16em] text-slate-400'>
                  {pendingRequest?.team} {pendingRequest?.role} · {requestAgeSeconds}s
                </span>
                <span className='text-[11px] font-medium uppercase tracking-[0.16em] text-slate-500'>
                  {streamedTokenCount > 0 ? `~${streamedTokenCount} tok streamed`
                  : hiddenThinkingUpdateCount > 0 ?
                    `hidden reasoning active · ${hiddenThinkingUpdateCount} update${hiddenThinkingUpdateCount === 1 ? '' : 's'}`
                  : 'awaiting first visible token'}
                </span>
                <span className='text-[11px] font-medium uppercase tracking-[0.16em] text-slate-500'>
                  {pendingRequest?.resumeMode === 'prefill' ? 'resume: saved prefill'
                  : pendingRequest?.resumeMode === 'prefill-disabled' ? 'resume: saved locally only'
                  : 'resume: clean slate'}
                </span>
              </div>
              <Loader2 className='size-4 animate-spin text-slate-200' />
              <button
                type='button'
                onClick={() => {
                  cancelActiveRequest();
                  setIdleWarningMessage(null);
                  setPendingRequest(null);
                  setStreamedTokenCount(0);
                  setIsGamePaused(true);
                  setAppState('ready_for_turn');
                }}
                className='flex items-center gap-1 rounded-full border border-amber-400/40 bg-amber-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-amber-100 transition hover:bg-amber-500/20'
              >
                <Square className='size-3 fill-current' />
                Pause
              </button>
              <button
                type='button'
                onClick={retryCurrentTurn}
                className='flex items-center gap-1 rounded-full border border-rose-400/40 bg-rose-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-rose-100 transition hover:bg-rose-500/20'
              >
                <RotateCcw className='size-3' />
                Retry
              </button>
            </div>
          </div>
        )}
        {isGamePaused && appState === 'ready_for_turn' && (
          <div className='sticky flex w-full justify-end p-2'>
            <div className='flex items-center gap-3 rounded-full border border-slate-500/40 bg-slate-900/90 px-4 py-2 text-xs font-semibold tracking-wide text-slate-100 shadow-lg'>
              <div className='flex flex-col text-right'>
                <span>{currentTurnPrefill ? 'Paused with saved prefill' : 'Paused'}</span>
                {currentTurnPrefill && (
                  <span className='text-[11px] font-medium uppercase tracking-[0.16em] text-slate-500'>
                    {gameState.agents[currentTurnPrefill.team][currentTurnPrefill.role]
                      .openRouterAssistantPrefillEnabled === false ?
                      'resume: saved locally only'
                    : 'resume: saved prefill'}
                  </span>
                )}
              </div>
              <Pause className='text-slate-200' />
              {currentTurnPrefill && (
                <button
                  type='button'
                  onClick={retryCurrentTurn}
                  className='flex items-center gap-1 rounded-full border border-rose-400/40 bg-rose-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-rose-100 transition hover:bg-rose-500/20'
                >
                  <RotateCcw className='size-3' />
                  Retry Fresh
                </button>
              )}
            </div>
          </div>
        )}
      </div>

      <SettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
    </div>
  );
}
