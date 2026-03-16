import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  type ReactNode,
} from "react";
import { Octokit } from "@octokit/rest";
import { createOctokit, validateToken } from "../lib/github-api";
import {
  exchangeAuthCode,
  fetchCurrentUser,
  refreshAccessToken,
  logoutApi,
} from "../lib/jiun-api";

interface GitHubState {
  octokit: Octokit | null;
  user: { login: string; avatar_url: string } | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  loginWithToken: (token: string) => Promise<void>;
  loginWithOAuth: (code: string) => Promise<void>;
  logout: () => void;
}

const GITHUB_TOKEN_KEY = "admin_github_token";

function saveToken(token: string) {
  try {
    sessionStorage.setItem(GITHUB_TOKEN_KEY, token);
  } catch {}
}

function getToken(): string | null {
  try {
    return sessionStorage.getItem(GITHUB_TOKEN_KEY);
  } catch {
    return null;
  }
}

function removeToken() {
  try {
    sessionStorage.removeItem(GITHUB_TOKEN_KEY);
    localStorage.removeItem(GITHUB_TOKEN_KEY);
  } catch {}
}

const GitHubContext = createContext<GitHubState | undefined>(undefined);

export function GitHubProvider({ children }: { children: ReactNode }) {
  const [octokit, setOctokit] = useState<Octokit | null>(null);
  const [user, setUser] = useState<{ login: string; avatar_url: string } | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const isAuthenticated = octokit !== null && user !== null;

  const authenticateWithGitHubToken = useCallback(async (token: string) => {
    const kit = createOctokit(token);
    const userData = await validateToken(kit);
    saveToken(token);
    setOctokit(kit);
    setUser({ login: userData.login, avatar_url: userData.avatarUrl });
  }, []);

  // Check for OAuth callback code in URL on mount
  useEffect(() => {
    let cancelled = false;

    const init = async () => {
      // Check for OAuth callback code
      const url = new URL(window.location.href);
      const code = url.searchParams.get("code");
      const oauthError = url.searchParams.get("error");

      if (oauthError) {
        setError(`OAuth error: ${oauthError}`);
        url.searchParams.delete("error");
        window.history.replaceState({}, "", url.toString());
        setIsLoading(false);
        return;
      }

      if (code) {
        // Clean URL
        url.searchParams.delete("code");
        window.history.replaceState({}, "", url.toString());

        try {
          const jwtToken = await exchangeAuthCode(code);
          const { githubToken } = await fetchCurrentUser(jwtToken);

          if (!githubToken) {
            throw new Error("GitHub token not available. Please ensure the OAuth app has repo scope.");
          }

          if (!cancelled) {
            await authenticateWithGitHubToken(githubToken);
          }
        } catch (err) {
          if (!cancelled) {
            setError(err instanceof Error ? err.message : "OAuth login failed");
          }
        }

        if (!cancelled) setIsLoading(false);
        return;
      }

      // Try restore from session storage
      const savedToken = getToken();
      if (savedToken) {
        try {
          if (!cancelled) await authenticateWithGitHubToken(savedToken);
        } catch {
          removeToken();
        }
      }

      // Try refresh via jiun-api cookie
      if (!savedToken) {
        try {
          const jwtToken = await refreshAccessToken();
          if (jwtToken && !cancelled) {
            const { githubToken } = await fetchCurrentUser(jwtToken);
            if (githubToken && !cancelled) {
              await authenticateWithGitHubToken(githubToken);
            }
          }
        } catch {
          // No valid session
        }
      }

      if (!cancelled) setIsLoading(false);
    };

    init();
    return () => { cancelled = true; };
  }, [authenticateWithGitHubToken]);

  const loginWithToken = useCallback(async (token: string) => {
    setIsLoading(true);
    setError(null);
    try {
      await authenticateWithGitHubToken(token);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setIsLoading(false);
    }
  }, [authenticateWithGitHubToken]);

  const loginWithOAuth = useCallback(async (code: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const jwtToken = await exchangeAuthCode(code);
      const { githubToken } = await fetchCurrentUser(jwtToken);
      if (!githubToken) {
        throw new Error("GitHub token not returned from API");
      }
      await authenticateWithGitHubToken(githubToken);
    } catch (err) {
      setError(err instanceof Error ? err.message : "OAuth login failed");
    } finally {
      setIsLoading(false);
    }
  }, [authenticateWithGitHubToken]);

  const logout = useCallback(() => {
    setOctokit(null);
    setUser(null);
    setError(null);
    removeToken();
    logoutApi();
  }, []);

  const value = useMemo<GitHubState>(() => ({
    octokit,
    user,
    isAuthenticated,
    isLoading,
    error,
    loginWithToken,
    loginWithOAuth,
    logout,
  }), [octokit, user, isAuthenticated, isLoading, error, loginWithToken, loginWithOAuth, logout]);

  return (
    <GitHubContext.Provider value={value}>{children}</GitHubContext.Provider>
  );
}

export function useGitHub(): GitHubState {
  const context = useContext(GitHubContext);
  if (context === undefined) {
    throw new Error("useGitHub must be used within a GitHubProvider");
  }
  return context;
}
