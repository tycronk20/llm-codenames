import { Loader2 } from 'lucide-react';
import { colorizeMessage } from '../utils/colors';
import { CardType, TeamColor } from '../utils/game';
import { LLMModel } from '../utils/models';

export type ChatMessage = {
  model: LLMModel;
  message: string;
  team: TeamColor;
  cards?: CardType[];
  isStreaming?: boolean;
};

export function Chat({ message, team, model, cards, isStreaming = false }: ChatMessage) {
  return (
    <div className='flex flex-col p-3'>
      {/* Model chat heading */}
      <div className='flex flex-row items-center gap-2'>
        {/* Avatar logo */}
        <div className='flex-shrink-0'>
          <img
            src={model.logo}
            alt={model.shortName}
            className={`h-6 w-6 rounded-full border-black p-1 ${
              team === 'red' ? 'bg-red-50' : 'bg-sky-50'
            }`}
          />
        </div>
        {/* Model name */}
        <span
          className={`text-sm font-semibold ${team === 'blue' ? 'text-sky-500' : 'text-rose-500'}`}
        >
          {model.shortName}
        </span>
        {isStreaming && (
          <span className='inline-flex items-center gap-1 rounded-full border border-slate-500/40 bg-slate-900/80 px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-300'>
            <Loader2 className='size-3 animate-spin' />
            CoT
          </span>
        )}
      </div>

      {/* Chat message */}
      <div className='flex flex-col'>
        {/* Colorize the words in the message based on game cards */}
        <p
          className={`mt-2 whitespace-pre-line border-l-4 pl-3 text-sm italic ${
            isStreaming ? 'opacity-90' : ''
          } text-slate-300 ${team === 'blue' ? 'border-sky-600/90' : 'border-rose-600/90'
          }`}
        >
          {cards ? colorizeMessage(message, cards) : message}
        </p>
      </div>
    </div>
  );
}
