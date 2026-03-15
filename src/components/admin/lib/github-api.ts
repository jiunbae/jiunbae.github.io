import { Octokit } from "@octokit/rest";

export const OWNER = "jiunbae";
export const REPO = "jiunbae.github.io";
export const BRANCH = "main";

const CONTENT_PREFIXES = [
  "src/content/posts/",
  "src/content/notes/",
  "src/content/reviews/",
];

export function createOctokit(token: string): Octokit {
  return new Octokit({ auth: token });
}

export async function validateToken(
  octokit: Octokit,
): Promise<{ login: string; name: string | null; avatarUrl: string }> {
  const { data } = await octokit.rest.users.getAuthenticated();
  return {
    login: data.login,
    name: data.name,
    avatarUrl: data.avatar_url,
  };
}

export interface TreeFile {
  path: string;
  sha: string;
  size: number;
}

export async function fetchRepoTree(
  octokit: Octokit,
  owner: string = OWNER,
  repo: string = REPO,
  branch: string = BRANCH,
): Promise<TreeFile[]> {
  const { data } = await octokit.rest.git.getTree({
    owner,
    repo,
    tree_sha: branch,
    recursive: "1",
  });

  return (data.tree ?? [])
    .filter(
      (item) =>
        item.type === "blob" &&
        item.path &&
        CONTENT_PREFIXES.some((prefix) => item.path!.startsWith(prefix)),
    )
    .map((item) => ({
      path: item.path!,
      sha: item.sha!,
      size: item.size ?? 0,
    }));
}

export interface FileContent {
  content: string;
  sha: string;
}

export async function fetchFileContent(
  octokit: Octokit,
  owner: string = OWNER,
  repo: string = REPO,
  path: string,
  branch: string = BRANCH,
): Promise<FileContent> {
  const { data } = await octokit.rest.repos.getContent({
    owner,
    repo,
    path,
    ref: branch,
  });

  if (Array.isArray(data) || data.type !== "file") {
    throw new Error(`Path "${path}" is not a file`);
  }

  const content = decodeBase64(data.content);

  return {
    content,
    sha: data.sha,
  };
}

export async function createOrUpdateFile(
  octokit: Octokit,
  owner: string = OWNER,
  repo: string = REPO,
  path: string,
  content: string,
  message: string,
  sha?: string,
): Promise<{ sha: string }> {
  const { data } = await octokit.rest.repos.createOrUpdateFileContents({
    owner,
    repo,
    path,
    message,
    content: encodeBase64(content),
    branch: BRANCH,
    ...(sha ? { sha } : {}),
  });

  return { sha: data.content?.sha ?? "" };
}

export async function deleteFile(
  octokit: Octokit,
  owner: string = OWNER,
  repo: string = REPO,
  path: string,
  sha: string,
  message: string,
): Promise<void> {
  await octokit.rest.repos.deleteFile({
    owner,
    repo,
    path,
    message,
    sha,
    branch: BRANCH,
  });
}

export async function uploadImage(
  octokit: Octokit,
  owner: string = OWNER,
  repo: string = REPO,
  path: string,
  base64Content: string,
  message: string,
): Promise<{ sha: string }> {
  const { data } = await octokit.rest.repos.createOrUpdateFileContents({
    owner,
    repo,
    path,
    message,
    content: base64Content,
    branch: BRANCH,
  });

  return { sha: data.content?.sha ?? "" };
}

function encodeBase64(str: string): string {
  if (typeof btoa === "function") {
    return btoa(unescape(encodeURIComponent(str)));
  }
  return Buffer.from(str, "utf-8").toString("base64");
}

function decodeBase64(encoded: string): string {
  const cleaned = encoded.replace(/\n/g, "");
  if (typeof atob === "function") {
    return decodeURIComponent(escape(atob(cleaned)));
  }
  return Buffer.from(cleaned, "base64").toString("utf-8");
}
