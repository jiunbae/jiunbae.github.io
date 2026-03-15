import { useState, useCallback } from "react";
import type { Draft } from "@/utils/storage";
import { useDrafts } from "../hooks/useDrafts";

interface DraftManagerProps {
  onLoad: (draft: Draft) => void;
  currentDraftId?: string;
}

export default function DraftManager({ onLoad, currentDraftId }: DraftManagerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const { drafts, deleteDraft, deleteAllDrafts, refreshDrafts } = useDrafts();

  const handleOpen = useCallback(() => {
    refreshDrafts();
    setIsOpen(true);
  }, [refreshDrafts]);

  const handleClose = useCallback(() => {
    setIsOpen(false);
  }, []);

  const handleLoad = useCallback(
    (draft: Draft) => {
      onLoad(draft);
      setIsOpen(false);
    },
    [onLoad],
  );

  const handleDelete = useCallback(
    (id: string) => {
      deleteDraft(id);
    },
    [deleteDraft],
  );

  const handleDeleteAll = useCallback(() => {
    deleteAllDrafts();
  }, [deleteAllDrafts]);

  return (
    <div className="draft-manager">
      <button type="button" className="btn-drafts" onClick={handleOpen}>
        Drafts{drafts.length > 0 && <span> ({drafts.length})</span>}
      </button>

      {isOpen && (
        <div className="draft-modal" onClick={handleClose}>
          <div
            className="draft-modal-content"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="draft-modal-header">
              <h3>Saved Drafts</h3>
              <div>
                {drafts.length > 0 && (
                  <button
                    type="button"
                    className="btn-delete-all"
                    onClick={handleDeleteAll}
                  >
                    Delete All
                  </button>
                )}
                <button
                  type="button"
                  className="btn-close"
                  onClick={handleClose}
                >
                  Close
                </button>
              </div>
            </div>

            <div className="draft-list">
              {drafts.length === 0 ? (
                <p>No saved drafts.</p>
              ) : (
                drafts.map((draft) => (
                  <div
                    key={draft.id}
                    className={`draft-item${draft.id === currentDraftId ? " active" : ""}`}
                  >
                    <div className="draft-item-content">
                      <strong>{draft.title || "Untitled"}</strong>
                      <div className="draft-item-meta">
                        <span>{draft.type}</span>
                        <span>
                          {new Date(draft.savedAt).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                    <div className="draft-item-actions">
                      <button
                        type="button"
                        className="btn-load"
                        onClick={() => handleLoad(draft)}
                      >
                        Load
                      </button>
                      <button
                        type="button"
                        className="btn-delete"
                        onClick={() => handleDelete(draft.id)}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
