import { useState, useRef, useEffect, useCallback } from "react";

/**
 * Manages a two-click delete confirmation with auto-reset timer.
 * The timer is properly cleaned up on unmount to prevent
 * state-update-on-unmounted-component warnings.
 */
export function useConfirmDelete() {
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  const requestConfirm = useCallback((expId: string) => {
    // Clear any existing timer before setting a new one
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
    }
    setConfirmDeleteId(expId);
    timerRef.current = setTimeout(() => {
      setConfirmDeleteId(null);
      timerRef.current = null;
    }, 3000);
  }, []);

  const clearConfirm = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setConfirmDeleteId(null);
  }, []);

  return { confirmDeleteId, requestConfirm, clearConfirm };
}
