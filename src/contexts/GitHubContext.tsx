/**
 * GitHub API 상태 관리 Context
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Octokit } from '@octokit/rest';
import { getGitHubToken, saveGitHubToken, removeGitHubToken } from '@/utils/storage';
import { createOctokit, validateToken } from '@/utils/github';

interface GitHubContextType {
  octokit: Octokit | null;
  token: string | null;
  isAuthenticated: boolean;
  isValidating: boolean;
  login: (token: string) => Promise<boolean>;
  logout: () => void;
}

const GitHubContext = createContext<GitHubContextType | undefined>(undefined);

export const GitHubProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [octokit, setOctokit] = useState<Octokit | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isValidating, setIsValidating] = useState(true);

  // 초기 로드 시 토큰 확인
  useEffect(() => {
    const initAuth = async () => {
      const savedToken = getGitHubToken();
      if (savedToken) {
        const isValid = await validateToken(savedToken);
        if (isValid) {
          setToken(savedToken);
          setOctokit(createOctokit(savedToken));
          setIsAuthenticated(true);
        } else {
          removeGitHubToken();
        }
      }
      setIsValidating(false);
    };

    initAuth();
  }, []);

  const login = async (newToken: string): Promise<boolean> => {
    setIsValidating(true);
    const isValid = await validateToken(newToken);

    if (isValid) {
      saveGitHubToken(newToken);
      setToken(newToken);
      setOctokit(createOctokit(newToken));
      setIsAuthenticated(true);
      setIsValidating(false);
      return true;
    }

    setIsValidating(false);
    return false;
  };

  const logout = () => {
    removeGitHubToken();
    setToken(null);
    setOctokit(null);
    setIsAuthenticated(false);
  };

  return (
    <GitHubContext.Provider
      value={{
        octokit,
        token,
        isAuthenticated,
        isValidating,
        login,
        logout,
      }}
    >
      {children}
    </GitHubContext.Provider>
  );
};

export const useGitHub = (): GitHubContextType => {
  const context = useContext(GitHubContext);
  if (!context) {
    throw new Error('useGitHub must be used within a GitHubProvider');
  }
  return context;
};
