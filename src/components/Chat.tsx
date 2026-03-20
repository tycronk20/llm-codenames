import { Loader2 } from 'lucide-react';
import { colorizeMessage } from '../utils/colors';
import { ChatMessage } from '../utils/game';

export function Chat({ message, team, model, cards, isStreaming = false }: ChatMessage) {
  const body = cards ? colorizeMessage(message, cards) : message;
  const showBodyPlaceholder = isStreaming && !message.trim();

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
          <Loader2 className='size-3.5 shrink-0 animate-spin text-slate-400' aria-hidden />
        )}
      </div>

      {/* Chat message */}
      <div className='flex flex-col'>
        {/* Colorize the words in the message based on game cards */}
        <p
          className={`mt-2 whitespace-pre-line border-l-4 pl-3 text-sm italic ${
            isStreaming ? 'opacity-90' : ''
          } text-slate-300 ${team === 'blue' ? 'border-sky-600/90' : 'border-rose-600/90'
          } ${showBodyPlaceholder ? 'min-h-[2.5rem]' : ''}`}
        >
          {showBodyPlaceholder ? '\u00a0' : body}
        </p>
      </div>
    </div>
  );
}
