import { useState, useRef, useCallback } from "react";

export type SpacesStatus = { type: "success" | "error"; message: string } | null;

/**
 * Manages a transient status message for Spaces save/restore operations.
 * Shows a message for 3 seconds then auto-clears.
 */
export function useSpacesStatus() {
  const [spacesStatus, setSpacesStatus] = useState<SpacesStatus>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showStatus = useCallback((type: "success" | "error", message: string) => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
    }
    setSpacesStatus({ type, message });
    timerRef.current = setTimeout(() => {
      setSpacesStatus(null);
      timerRef.current = null;
    }, 3000);
  }, []);

  const clearStatus = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setSpacesStatus(null);
  }, []);

  return { spacesStatus, showStatus, clearStatus };
}
