import { useState, useCallback } from "react";
import {
  type Draft,
  getDrafts,
  saveDraft as storageSaveDraft,
  deleteDraft as storageDeleteDraft,
  getDraft as storageGetDraft,
  deleteAllDrafts as storageDeleteAllDrafts,
} from "@/utils/storage";

export type { Draft };

export interface UseDrafts {
  drafts: Draft[];
  saveDraft: (draft: Draft) => void;
  deleteDraft: (id: string) => void;
  getDraft: (id: string) => Draft | null;
  deleteAllDrafts: () => void;
  refreshDrafts: () => void;
}

export function useDrafts(): UseDrafts {
  const [drafts, setDrafts] = useState<Draft[]>(() => getDrafts());

  const refreshDrafts = useCallback(() => {
    setDrafts(getDrafts());
  }, []);

  const saveDraft = useCallback((draft: Draft) => {
    storageSaveDraft(draft);
    setDrafts(getDrafts());
  }, []);

  const deleteDraft = useCallback((id: string) => {
    storageDeleteDraft(id);
    setDrafts(getDrafts());
  }, []);

  const getDraftById = useCallback((id: string): Draft | null => {
    return storageGetDraft(id);
  }, []);

  const deleteAllDrafts = useCallback(() => {
    storageDeleteAllDrafts();
    setDrafts([]);
  }, []);

  return {
    drafts,
    saveDraft,
    deleteDraft,
    getDraft: getDraftById,
    deleteAllDrafts,
    refreshDrafts,
  };
}
