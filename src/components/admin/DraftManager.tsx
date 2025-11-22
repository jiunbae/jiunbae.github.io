/**
 * Draft 관리 컴포넌트
 * 임시 저장된 글 목록 및 불러오기/삭제
 */

import React, { useState, useEffect } from 'react';
import { getDrafts, deleteDraft, Draft } from '@/utils/storage';
import { format } from 'date-fns';

interface DraftManagerProps {
  onLoadDraft: (draft: Draft) => void;
  currentDraftId?: string;
}

const DraftManager: React.FC<DraftManagerProps> = ({ onLoadDraft, currentDraftId }) => {
  const [drafts, setDrafts] = useState<Draft[]>([]);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    loadDrafts();
  }, []);

  const loadDrafts = () => {
    setDrafts(getDrafts());
  };

  const handleDeleteDraft = (id: string) => {
    if (confirm('이 임시 저장을 삭제하시겠습니까?')) {
      deleteDraft(id);
      loadDrafts();
    }
  };

  const handleLoadDraft = (draft: Draft) => {
    if (
      currentDraftId !== draft.id &&
      confirm('현재 작성 중인 내용을 임시 저장으로 덮어쓰시겠습니까?')
    ) {
      onLoadDraft(draft);
      setIsOpen(false);
    } else if (currentDraftId === draft.id) {
      onLoadDraft(draft);
      setIsOpen(false);
    }
  };

  if (drafts.length === 0) {
    return null;
  }

  return (
    <div className="draft-manager">
      <button onClick={() => setIsOpen(!isOpen)} className="btn-drafts">
        임시 저장 ({drafts.length})
      </button>

      {isOpen && (
        <div className="draft-modal">
          <div className="draft-modal-content">
            <div className="draft-modal-header">
              <h3>임시 저장 목록</h3>
              <button onClick={() => setIsOpen(false)} className="btn-close">
                ✕
              </button>
            </div>

            <ul className="draft-list">
              {drafts.map((draft) => (
                <li key={draft.id} className="draft-item">
                  <div className="draft-item-content">
                    <h4>{draft.title || '(제목 없음)'}</h4>
                    <p className="draft-item-meta">
                      {format(new Date(draft.savedAt), 'yyyy-MM-dd HH:mm:ss')}
                      {' • '}
                      {draft.type === 'post' ? '포스트' : '노트'}
                    </p>
                  </div>
                  <div className="draft-item-actions">
                    <button
                      onClick={() => handleLoadDraft(draft)}
                      className="btn-load"
                    >
                      불러오기
                    </button>
                    <button
                      onClick={() => handleDeleteDraft(draft.id)}
                      className="btn-delete"
                    >
                      삭제
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default DraftManager;
