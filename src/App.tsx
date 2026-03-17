import { Loader2, Pause, Play, Square } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import Card from './components/Card';
import { Chat } from './components/Chat';
import { Scoreboard } from './components/Scoreboard';
import {
  GameState,
  initializeGameState,
  OperativeMove,
  SpymasterMove,
  updateGameStateFromOperativeMove,
  updateGameStateFromSpymasterMove,
} from './utils/game';
import { createMessagesFromGameState, streamLLMResponse } from './utils/llm';

type AppState = 'game_start' | 'ready_for_turn' | 'waiting_for_response' | 'error' | 'game_over';
type PendingRequest = {
  modelName: string;
  team: GameState['currentTeam'];
  role: GameState['currentRole'];
  startedAt: number;
};

function estimateTokenCount(text: string) {
  const matches = text.match(/\w+|[^\s\w]/g);
  return matches?.length ?? 0;
}

export default function App() {
  const [gameState, setGameState] = useState<GameState>(initializeGameState());
  const [appState, setAppState] = useState<AppState>('game_start');
  const [isGamePaused, setIsGamePaused] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [idleWarningMessage, setIdleWarningMessage] = useState<string | null>(null);
  const [pendingRequest, setPendingRequest] = useState<PendingRequest | null>(null);
  const [pendingChatMessage, setPendingChatMessage] = useState<{
    team: PendingRequest['team'];
    role: PendingRequest['role'];
    reasoning: string;
  } | null>(null);
  const [streamedTokenCount, setStreamedTokenCount] = useState(0);
  const [requestAgeSeconds, setRequestAgeSeconds] = useState(0);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const activeTurnKeyRef = useRef<string | null>(null);
  const activeRequestControllerRef = useRef<AbortController | null>(null);

  const cancelActiveRequest = () => {
    activeRequestControllerRef.current?.abort();
    activeRequestControllerRef.current = null;
    activeTurnKeyRef.current = null;
  };

  useEffect(() => {
    if (isGamePaused) {
      cancelActiveRequest();
      setIdleWarningMessage(null);
      setPendingRequest(null);
      setPendingChatMessage(null);
      setStreamedTokenCount(0);
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
      setIsGamePaused(true);
      setAppState('game_over');
      return;
    }

    if (appState === 'ready_for_turn') {
      const turnKey = [
        gameState.chatHistory.length,
        gameState.currentTeam,
        gameState.currentRole,
        gameState.remainingRed,
        gameState.remainingBlue,
      ].join(':');

      if (activeTurnKeyRef.current === turnKey) {
        return;
      }

      const turnState = gameState;
      const activeModel = turnState.agents[turnState.currentTeam][turnState.currentRole];
      const controller = new AbortController();
      activeTurnKeyRef.current = turnKey;
      activeRequestControllerRef.current = controller;
      setAppState('waiting_for_response');
      setPendingRequest({
        modelName: activeModel.modelName,
        team: turnState.currentTeam,
        role: turnState.currentRole,
        startedAt: Date.now(),
      });
      setPendingChatMessage(null);
      setErrorMessage(null);
      setIdleWarningMessage(null);
      setStreamedTokenCount(0);

      void (async () => {
        try {
          let completedMove: SpymasterMove | OperativeMove | null = null;
          for await (const update of streamLLMResponse(
            {
              messages: createMessagesFromGameState(turnState),
              modelId: activeModel.id,
              role: turnState.currentRole,
            },
            controller.signal,
            {
              onIdleStateChange: (message) => {
                if (activeRequestControllerRef.current !== controller || controller.signal.aborted) {
                  return;
                }

                setIdleWarningMessage(message);
              },
            },
          )) {
            if (activeRequestControllerRef.current !== controller || controller.signal.aborted) {
              return;
            }

            if (update.type === 'reasoning') {
              setPendingChatMessage({
                team: turnState.currentTeam,
                role: turnState.currentRole,
                reasoning: update.reasoning,
              });
              setStreamedTokenCount(estimateTokenCount(update.reasoning));
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
  }, [appState, gameState, isGamePaused]);

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
    <div className='flex min-h-screen flex-col items-center justify-around gap-2 bg-gradient-to-br from-slate-800 to-slate-600 antialiased lg:flex-row'>
      {/* Left panel: Scoreboard + Game Board + Game Controls */}
      <div className='flex w-full flex-col items-center gap-y-6 sm:mt-4 lg:w-2/3 lg:gap-y-8'>
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
              setIdleWarningMessage(null);
              setPendingRequest(null);
              setPendingChatMessage(null);
              setStreamedTokenCount(0);
            }
            if (appState === 'game_over' || appState === 'error') {
              setGameState(initializeGameState());
              setAppState('game_start');
              setErrorMessage(null);
              setIdleWarningMessage(null);
              setPendingRequest(null);
              setPendingChatMessage(null);
              setStreamedTokenCount(0);
            }
            setIsGamePaused(nextPausedState);
          }}
          className='mb-6 flex w-36 items-center justify-center gap-2 rounded bg-slate-200 px-2 py-2 font-bold text-slate-800 hover:bg-slate-300'
        >
          {appState === 'game_start' || appState === 'game_over' || appState === 'error' ?
            <>
              <Play className='inline size-4' /> New Game
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
        className='relative w-full max-w-4xl bg-slate-800/50 p-2 backdrop-blur-sm md:h-screen md:overflow-y-auto lg:w-1/3 lg:border-l lg:border-slate-500/30'
      >
        {gameState.chatHistory.map((message, index) => (
          <Chat key={index} {...message} />
        ))}
        {pendingChatMessage && pendingRequest && (
          <Chat
            message={pendingChatMessage.reasoning}
            model={gameState.agents[pendingChatMessage.team][pendingChatMessage.role]}
            team={pendingChatMessage.team}
            cards={gameState.cards}
            isStreaming={true}
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
                  ~{streamedTokenCount} tok streamed
                </span>
              </div>
              <Loader2 className='size-4 animate-spin text-slate-200' />
              <button
                type='button'
                onClick={() => {
                  cancelActiveRequest();
                  setIdleWarningMessage(null);
                  setPendingRequest(null);
                  setPendingChatMessage(null);
                  setStreamedTokenCount(0);
                  setIsGamePaused(true);
                  setAppState('ready_for_turn');
                }}
                className='flex items-center gap-1 rounded-full border border-rose-400/40 bg-rose-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-rose-100 transition hover:bg-rose-500/20'
              >
                <Square className='size-3 fill-current' />
                Stop
              </button>
            </div>
          </div>
        )}
        {isGamePaused && appState === 'ready_for_turn' && (
          <div className='sticky flex w-full justify-end p-2'>
            <Pause className='text-slate-200' />
          </div>
        )}
      </div>
    </div>
  );
}
