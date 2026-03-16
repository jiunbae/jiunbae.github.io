/// <reference types="astro/client" />

interface ImportMetaEnv {
  readonly PUBLIC_GITHUB_OWNER: string;
  readonly PUBLIC_GITHUB_REPO: string;
  readonly PUBLIC_GITHUB_BRANCH: string;
  readonly PUBLIC_API_BASE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
