import { useState, useRef, useCallback } from "react";
import { useGitHub as useGitHubContext } from "../context/GitHubContext";
import {
  OWNER,
  REPO,
  BRANCH,
  fetchRepoTree,
  fetchFileContent,
  createOrUpdateFile,
  deleteFile,
  uploadImage as uploadImageAPI,
} from "../lib/github-api";

export interface TreeEntry {
  path: string;
  type: string;
  sha: string;
}

export interface UseGitHubAPI {
  loading: boolean;
  error: string | null;
  tree: TreeEntry[] | null;
  contentCache: Map<string, { content: string; sha: string }>;
  fetchTree: () => Promise<void>;
  fetchContent: (path: string) => Promise<{ content: string; sha: string }>;
  saveContent: (
    path: string,
    content: string,
    message: string,
    sha?: string,
  ) => Promise<void>;
  deleteContent: (
    path: string,
    sha: string,
    message: string,
  ) => Promise<void>;
  uploadImage: (
    path: string,
    base64: string,
    message: string,
  ) => Promise<string>;
}

export function useGitHubAPI(): UseGitHubAPI {
  const { octokit } = useGitHubContext();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tree, setTree] = useState<TreeEntry[] | null>(null);
  const contentCacheRef = useRef<Map<string, { content: string; sha: string }>>(
    new Map(),
  );

  const ensureOctokit = useCallback(() => {
    if (!octokit) {
      throw new Error("Not authenticated. Please log in first.");
    }
    return octokit;
  }, [octokit]);

  const fetchTree = useCallback(async () => {
    const kit = ensureOctokit();
    setLoading(true);
    setError(null);

    try {
      const files = await fetchRepoTree(kit, OWNER, REPO, BRANCH);
      const entries: TreeEntry[] = files.map((f) => ({
        path: f.path,
        type: "blob",
        sha: f.sha,
      }));
      setTree(entries);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to fetch tree";
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [ensureOctokit]);

  const fetchContent = useCallback(
    async (path: string) => {
      const cached = contentCacheRef.current.get(path);
      if (cached) return cached;

      const kit = ensureOctokit();
      setLoading(true);
      setError(null);

      try {
        const result = await fetchFileContent(kit, OWNER, REPO, path, BRANCH);
        const entry = { content: result.content, sha: result.sha };
        contentCacheRef.current.set(path, entry);
        return entry;
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to fetch content";
        setError(message);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [ensureOctokit],
  );

  const saveContent = useCallback(
    async (
      path: string,
      content: string,
      message: string,
      sha?: string,
    ) => {
      const kit = ensureOctokit();
      setLoading(true);
      setError(null);

      try {
        const result = await createOrUpdateFile(
          kit,
          OWNER,
          REPO,
          path,
          content,
          message,
          sha,
        );
        // Update cache with new sha
        contentCacheRef.current.set(path, { content, sha: result.sha });
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to save content";
        setError(errorMessage);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [ensureOctokit],
  );

  const deleteContent = useCallback(
    async (path: string, sha: string, message: string) => {
      const kit = ensureOctokit();
      setLoading(true);
      setError(null);

      try {
        await deleteFile(kit, OWNER, REPO, path, sha, message);
        contentCacheRef.current.delete(path);
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Failed to delete content";
        setError(msg);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [ensureOctokit],
  );

  const uploadImage = useCallback(
    async (path: string, base64: string, message: string): Promise<string> => {
      const kit = ensureOctokit();
      setLoading(true);
      setError(null);

      try {
        const result = await uploadImageAPI(
          kit,
          OWNER,
          REPO,
          path,
          base64,
          message,
        );
        return result.sha;
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Failed to upload image";
        setError(msg);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [ensureOctokit],
  );

  return {
    loading,
    error,
    tree,
    contentCache: contentCacheRef.current,
    fetchTree,
    fetchContent,
    saveContent,
    deleteContent,
    uploadImage,
  };
}
