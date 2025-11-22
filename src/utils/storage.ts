/**
 * localStorage 헬퍼 함수
 * GitHub PAT 및 Draft 데이터를 안전하게 저장/조회
 */

const GITHUB_TOKEN_KEY = 'github_pat';
const DRAFTS_KEY = 'blog_drafts';

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
  if (typeof window !== 'undefined') {
    localStorage.setItem(GITHUB_TOKEN_KEY, token);
  }
};

/**
 * GitHub Personal Access Token 조회
 */
export const getGitHubToken = (): string | null => {
  if (typeof window !== 'undefined') {
    return localStorage.getItem(GITHUB_TOKEN_KEY);
  }
  return null;
};

/**
 * GitHub Personal Access Token 삭제 (로그아웃)
 */
export const removeGitHubToken = (): void => {
  if (typeof window !== 'undefined') {
    localStorage.removeItem(GITHUB_TOKEN_KEY);
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
