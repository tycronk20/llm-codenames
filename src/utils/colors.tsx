import { CardType } from './game';

export const bgColorMap = {
  blue: 'bg-sky-700',
  red: 'bg-rose-600',
  black: 'bg-black',
  neutral: 'bg-slate-700',
};

export const borderColorMap = {
  blue: 'border-sky-600',
  red: 'border-rose-500',
  black: 'border-black',
  neutral: 'border-slate-700',
};

function normalizeTokenForHighlight(token: string): string {
  return token.replace(/[^a-zA-Z0-9]/g, '').toUpperCase();
}

const colorClasses = {
  red: 'text-rose-50 bg-rose-600/70',
  blue: 'text-sky-50 bg-sky-700/70',
  black: 'text-slate-50 bg-slate-800/90',
  neutral: 'text-slate-600 bg-orange-200/70',
} as const;

// Helper function to colorize words in the message based on the cards
export const colorizeMessage = (text: string, cards: CardType[]) => {
  const parts = text.split(/(\s+)/);
  const wordIndices: number[] = [];
  parts.forEach((part, i) => {
    if (part.trim()) wordIndices.push(i);
  });

  const cardPhrases = cards.map((c) => ({
    card: c,
    tokens: c.word.split(/\s+/).map((t) => normalizeTokenForHighlight(t)),
  }));
  cardPhrases.sort((a, b) => b.tokens.length - a.tokens.length);

  const ranges: { start: number; end: number; card: CardType; displayText: string }[] = [];

  for (let wi = 0; wi < wordIndices.length; ) {
    let matchedLen = 0;
    for (const { card, tokens } of cardPhrases) {
      if (tokens.length === 0 || wi + tokens.length > wordIndices.length) continue;
      let ok = true;
      for (let t = 0; t < tokens.length; t++) {
        const partIdx = wordIndices[wi + t];
        if (normalizeTokenForHighlight(parts[partIdx]) !== tokens[t]) {
          ok = false;
          break;
        }
      }
      if (ok) {
        const start = wordIndices[wi];
        const end = wordIndices[wi + tokens.length - 1];
        ranges.push({
          start,
          end,
          card,
          displayText: parts.slice(start, end + 1).join(''),
        });
        matchedLen = tokens.length;
        break;
      }
    }
    wi += matchedLen || 1;
  }

  const out: React.ReactNode[] = [];
  let i = 0;
  while (i < parts.length) {
    const range = ranges.find((r) => r.start === i);
    if (range) {
      out.push(
        <span
          key={`${range.start}-${range.end}`}
          className={`${colorClasses[range.card.color]} rounded px-1 font-semibold`}
        >
          {range.displayText}
        </span>,
      );
      i = range.end + 1;
      continue;
    }
    const part = parts[i];
    if (!part.trim()) {
      out.push(part);
    } else {
      out.push(part + ' ');
    }
    i += 1;
  }
  return out;
};
