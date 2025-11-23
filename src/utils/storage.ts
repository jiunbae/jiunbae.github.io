/**
 * localStorage/sessionStorage 헬퍼 함수
 * GitHub PAT 및 Draft 데이터를 저장/조회
 *
 * SECURITY NOTE:
 * - GitHub Personal Access Token은 AES-GCM 암호화되어 저장됩니다
 * - Web Crypto API를 사용한 브라우저 기반 암호화
 * - localStorage/sessionStorage는 XSS 공격에 취약하므로 주의가 필요합니다
 * - 공용 컴퓨터에서는 sessionStorage 사용을 권장합니다
 * - 사용 후 반드시 로그아웃하여 토큰을 제거하세요
 */

import { encrypt, decrypt, isValidEncryptedData } from './crypto';

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
  type: 'post' | 'note' | 'review';
  savedAt: string;
}

/**
 * GitHub Personal Access Token 저장 (암호화됨)
 */
export const saveGitHubToken = async (token: string): Promise<void> => {
  const storage = getStorage();
  if (storage) {
    try {
      const encrypted = await encrypt(token);
      storage.setItem(GITHUB_TOKEN_KEY, encrypted);
    } catch (error) {
      console.error('Failed to encrypt token:', error);
      // 암호화 실패 시 저장하지 않음 (보안상 평문 저장 방지)
      throw new Error('토큰 암호화에 실패했습니다');
    }
  }
};

/**
 * GitHub Personal Access Token 조회 (복호화됨)
 */
export const getGitHubToken = async (): Promise<string | null> => {
  const storage = getStorage();
  if (storage) {
    const encryptedToken = storage.getItem(GITHUB_TOKEN_KEY);
    if (encryptedToken) {
      try {
        // 암호화된 데이터인지 확인
        if (!isValidEncryptedData(encryptedToken)) {
          console.warn('Invalid encrypted token format, removing...');
          removeGitHubToken();
          return null;
        }

        const decrypted = await decrypt(encryptedToken);
        return decrypted;
      } catch (error) {
        console.error('Failed to decrypt token:', error);
        // 복호화 실패 시 토큰 제거 (손상된 데이터)
        removeGitHubToken();
        return null;
      }
    }
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
