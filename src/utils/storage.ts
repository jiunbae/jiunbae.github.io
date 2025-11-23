/**
 * localStorage/sessionStorage 헬퍼 함수
 * GitHub PAT 및 Draft 데이터를 저장/조회
 *
 * SECURITY WARNING:
 * - localStorage/sessionStorage는 XSS 공격에 취약합니다
 * - Personal Access Token을 저장할 때는 보안 위험을 인지하고 사용하세요
 * - 공용 컴퓨터에서는 sessionStorage 사용을 권장합니다
 * - 사용 후 반드시 로그아웃하여 토큰을 제거하세요
 */

const GITHUB_TOKEN_KEY = 'github_pat';
const DRAFTS_KEY = 'blog_drafts';
const USE_SESSION_STORAGE_KEY = 'use_session_storage';

/**
 * 스토리지 타입 선택 (localStorage 또는 sessionStorage)
 */
const getStorage = (): Storage | null => {
  if (typeof window === 'undefined') return null;

  const useSessionStorage = localStorage.getItem(USE_SESSION_STORAGE_KEY) === 'true';
  return useSessionStorage ? sessionStorage : localStorage;
};

/**
 * 스토리지 타입 설정
 */
export const setStorageType = (useSessionStorage: boolean): void => {
  if (typeof window !== 'undefined') {
    localStorage.setItem(USE_SESSION_STORAGE_KEY, String(useSessionStorage));
  }
};

/**
 * 현재 스토리지 타입 확인
 */
export const isUsingSessionStorage = (): boolean => {
  if (typeof window === 'undefined') return false;
  return localStorage.getItem(USE_SESSION_STORAGE_KEY) === 'true';
};

export interface Draft {
  id: string;
  title: string;
  content: string;
  frontmatter: {
    title: string;
    description: string;
    date: string;
    slug: string;
    tags: string[];
    heroImage?: string;
    heroImageAlt?: string;
  };
  type: 'post' | 'note';
  savedAt: string;
}

/**
 * GitHub Personal Access Token 저장
 */
export const saveGitHubToken = (token: string): void => {
  const storage = getStorage();
  if (storage) {
    storage.setItem(GITHUB_TOKEN_KEY, token);
  }
};

/**
 * GitHub Personal Access Token 조회
 */
export const getGitHubToken = (): string | null => {
  const storage = getStorage();
  if (storage) {
    return storage.getItem(GITHUB_TOKEN_KEY);
  }
  return null;
};

/**
 * GitHub Personal Access Token 삭제 (로그아웃)
 */
export const removeGitHubToken = (): void => {
  // Both storages should be cleared to ensure complete logout
  if (typeof window !== 'undefined') {
    localStorage.removeItem(GITHUB_TOKEN_KEY);
    sessionStorage.removeItem(GITHUB_TOKEN_KEY);
  }
};

/**
 * Draft 목록 조회
 */
export const getDrafts = (): Draft[] => {
  if (typeof window !== 'undefined') {
    const draftsJson = localStorage.getItem(DRAFTS_KEY);
    if (draftsJson) {
      try {
        return JSON.parse(draftsJson);
      } catch (e) {
        console.error('Failed to parse drafts:', e);
        return [];
      }
    }
  }
  return [];
};

/**
 * Draft 저장
 */
export const saveDraft = (draft: Draft): void => {
  if (typeof window !== 'undefined') {
    const drafts = getDrafts();
    const existingIndex = drafts.findIndex((d) => d.id === draft.id);

    if (existingIndex >= 0) {
      drafts[existingIndex] = draft;
    } else {
      drafts.push(draft);
    }

    localStorage.setItem(DRAFTS_KEY, JSON.stringify(drafts));
  }
};

/**
 * Draft 삭제
 */
export const deleteDraft = (id: string): void => {
  if (typeof window !== 'undefined') {
    const drafts = getDrafts().filter((d) => d.id !== id);
    localStorage.setItem(DRAFTS_KEY, JSON.stringify(drafts));
  }
};

/**
 * 특정 Draft 조회
 */
export const getDraft = (id: string): Draft | null => {
  const drafts = getDrafts();
  return drafts.find((d) => d.id === id) || null;
};

/**
 * 모든 Draft 삭제
 */
export const deleteAllDrafts = (): void => {
  if (typeof window !== 'undefined') {
    localStorage.removeItem(DRAFTS_KEY);
  }
};
