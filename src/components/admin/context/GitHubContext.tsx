import {
  createContext,
  useContext,
  useState,
  useEffect,
  useRef,
  useCallback,
  type ReactNode,
} from "react";
import { Octokit } from "@octokit/rest";
import { createOctokit, validateToken } from "../lib/github-api";
import {
  saveGitHubToken,
  getGitHubToken,
  removeGitHubToken,
} from "@/utils/storage";
import { setPassphrase, clearPassphrase } from "@/utils/crypto";

interface GitHubState {
  octokit: Octokit | null;
  user: { login: string; avatar_url: string } | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (token: string, passphrase: string) => Promise<void>;
  logout: () => void;
}

const GitHubContext = createContext<GitHubState | undefined>(undefined);

export function GitHubProvider({ children }: { children: ReactNode }) {
  const [octokit, setOctokit] = useState<Octokit | null>(null);
  const [user, setUser] = useState<{ login: string; avatar_url: string } | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const passphraseRef = useRef<string>("");

  const isAuthenticated = octokit !== null && user !== null;

  // Restore token from storage on mount
  useEffect(() => {
    let cancelled = false;

    const restore = async () => {
      try {
        const token = await getGitHubToken();
        if (!token || cancelled) {
          setIsLoading(false);
          return;
        }

        const kit = createOctokit(token);
        const userData = await validateToken(kit);

        if (!cancelled) {
          setOctokit(kit);
          setUser({ login: userData.login, avatar_url: userData.avatarUrl });
        }
      } catch {
        // Token expired or invalid — silently clear it
        removeGitHubToken();
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    };

    restore();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (token: string, passphrase: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const kit = createOctokit(token);
      const userData = await validateToken(kit);

      passphraseRef.current = passphrase;
      if (passphrase) setPassphrase(passphrase);
      await saveGitHubToken(token);

      setOctokit(kit);
      setUser({ login: userData.login, avatar_url: userData.avatarUrl });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Authentication failed";
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    setOctokit(null);
    setUser(null);
    setError(null);
    passphraseRef.current = "";
    clearPassphrase();
    removeGitHubToken();
  }, []);

  const value: GitHubState = {
    octokit,
    user,
    isAuthenticated,
    isLoading,
    error,
    login,
    logout,
  };

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
