import { useState, useEffect, useRef, useCallback } from "react";
import { useDrafts } from "./useDrafts";

interface AutoSaveData {
  title: string;
  content: string;
  frontmatter: any;
  type: "post" | "note" | "review";
}

interface AutoSaveResult {
  lastSaved: Date | null;
  isSaving: boolean;
}

export function useAutoSave(
  data: AutoSaveData,
  draftId: string,
  enabled: boolean,
  delay: number = 5000,
): AutoSaveResult {
  const { saveDraft } = useDrafts();
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dataRef = useRef(data);
  const draftIdRef = useRef(draftId);

  // Keep refs up to date
  dataRef.current = data;
  draftIdRef.current = draftId;

  const performSave = useCallback(() => {
    const current = dataRef.current;
    const id = draftIdRef.current;

    // Skip saving if there is no meaningful content
    if (!current.title && !current.content) return;

    setIsSaving(true);

    try {
      saveDraft({
        id,
        title: current.title,
        content: current.content,
        frontmatter: current.frontmatter,
        type: current.type,
        savedAt: new Date().toISOString(),
      });
      setLastSaved(new Date());
    } finally {
      setIsSaving(false);
    }
  }, [saveDraft]);

  // Debounced auto-save — only trigger on title/content changes (primitives)
  // Frontmatter changes are captured via the ref on next save cycle
  useEffect(() => {
    if (!enabled) return;

    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }

    timerRef.current = setTimeout(() => {
      performSave();
    }, delay);

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [data.title, data.content, data.type, enabled, delay, performSave]);

  return { lastSaved, isSaving };
}
