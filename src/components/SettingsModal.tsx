import { Loader2, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { allModelCatalog, MIN_ACTIVE_MODELS } from '../utils/modelCatalog';
import {
  ACTIVE_MODEL_IDS_UPDATED_EVENT,
  getActiveModelIds,
  saveActiveModelIds,
} from '../utils/models';

type SettingsModalProps = {
  isOpen: boolean;
  onClose: () => void;
};

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [activeModelIds, setActiveModelIds] = useState<string[]>(() => getActiveModelIds());
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    setActiveModelIds(getActiveModelIds());
    setError(null);
  }, [isOpen]);

  useEffect(() => {
    const syncActiveModelIds = (event: Event) => {
      const detail = (event as CustomEvent<string[]>).detail;
      setActiveModelIds(Array.isArray(detail) ? detail : getActiveModelIds());
    };

    window.addEventListener(ACTIVE_MODEL_IDS_UPDATED_EVENT, syncActiveModelIds as EventListener);
    return () => {
      window.removeEventListener(ACTIVE_MODEL_IDS_UPDATED_EVENT, syncActiveModelIds as EventListener);
    };
  }, []);

  if (!isOpen) return null;

  const handleToggle = (id: string) => {
    setActiveModelIds((current) =>
      current.includes(id) ? current.filter((m) => m !== id) : [...current, id]
    );
  };

  const handleSave = async () => {
    if (activeModelIds.length < MIN_ACTIVE_MODELS) {
      setError(`At least ${MIN_ACTIVE_MODELS} models must be active.`);
      return;
    }

    try {
      setIsSaving(true);
      setError(null);
      saveActiveModelIds(activeModelIds);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className='fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm'>
      <div className='flex max-h-full w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-slate-700 bg-slate-900 shadow-2xl'>
        <div className='flex items-center justify-between border-b border-slate-700 bg-slate-800/50 p-4'>
          <h2 className='text-lg font-semibold text-slate-200'>Settings</h2>
          <button
            onClick={onClose}
            className='rounded-md p-1 text-slate-400 transition hover:bg-slate-700 hover:text-slate-200'
          >
            <X className='size-5' />
          </button>
        </div>

        <div className='flex-1 overflow-y-auto p-4'>
          <h3 className='mb-3 text-sm font-semibold uppercase tracking-wider text-slate-400'>
            Active Models
          </h3>
          {error && (
            <div className='mb-4 rounded-md border border-red-500/50 bg-red-500/10 p-3 text-sm text-red-400'>
              {error}
            </div>
          )}
          <div className='grid gap-2 sm:grid-cols-2'>
            {allModelCatalog.map((model) => {
              const isActive = activeModelIds.includes(model.id);
              return (
                <label
                  key={model.id}
                  className={`flex cursor-pointer items-center gap-3 rounded-lg border p-3 transition ${
                    isActive
                      ? 'border-emerald-500/50 bg-emerald-500/10'
                      : 'border-slate-700 bg-slate-800/50 hover:border-slate-600 hover:bg-slate-800'
                  }`}
                >
                  <input
                    type='checkbox'
                    className='sr-only'
                    checked={isActive}
                    onChange={() => handleToggle(model.id)}
                  />
                  <div
                    className={`flex size-5 shrink-0 items-center justify-center rounded border ${
                      isActive
                        ? 'border-emerald-500 bg-emerald-500'
                        : 'border-slate-500 bg-transparent'
                    }`}
                  >
                    {isActive && <div className='size-2.5 rounded-sm bg-white' />}
                  </div>
                  <div>
                    <div className='text-sm font-medium text-slate-200'>{model.modelName}</div>
                    <div className='text-xs text-slate-400'>{model.provider}</div>
                  </div>
                </label>
              );
            })}
          </div>
        </div>

        <div className='flex items-center justify-end border-t border-slate-700 bg-slate-800/50 p-4'>
          <button
            onClick={onClose}
            className='mr-3 px-4 py-2 text-sm font-medium text-slate-300 transition hover:text-white'
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className='flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-500 disabled:opacity-50'
          >
            {isSaving ? (
              <>
                <Loader2 className='size-4 animate-spin' />
                Saving...
              </>
            ) : (
              'Save Configuration'
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
