# Claude Architecture & Security Review (R02)

Reviewer: Claude Opus 4.6 (1M context)
Date: 2026-03-16
Scope: `src/components/admin/`, `src/pages/admin.astro`, `src/utils/crypto.ts`, `src/utils/storage.ts`

---

## Critical Findings (must fix)

### C1. GitHub PAT stored in localStorage is vulnerable to XSS token theft
**Files:** `src/utils/storage.ts`, `src/utils/crypto.ts`
**Category:** OWASP A03 (Injection / XSS), OWASP A07 (Identification and Authentication Failures)

The GitHub PAT is encrypted with AES-GCM and stored in localStorage. However, the encryption key is derived from the browser fingerprint (`navigator.userAgent`, `navigator.language`, screen dimensions, timezone offset) -- all values trivially accessible to any JavaScript running on the page. If an attacker achieves XSS (e.g., through the markdown preview, a compromised dependency, or any other vector), they can:

1. Read the ciphertext from `localStorage.getItem('github_pat')`
2. Reconstruct the identical fingerprint (they are running in the same browser)
3. Call `decrypt()` with the same key material to recover the plaintext PAT

This makes the encryption **security theater** against the primary threat (XSS). An optional passphrase is supported but the UI labels it "(optional)" and the code falls through to the fingerprint if no passphrase is set (`getEncryptionKey()` at line 120-123 of `crypto.ts`).

**Recommendation:** At minimum, require the passphrase (do not fall back to fingerprint). Better: use `sessionStorage` by default so the token does not survive tab close, and consider an OAuth flow that issues short-lived tokens instead of storing long-lived PATs.

### C2. Hardcoded PBKDF2 salt eliminates salt's purpose
**File:** `src/utils/crypto.ts:73`
**Category:** OWASP A02 (Cryptographic Failures)

```typescript
salt: encoder.encode('jiunbae-blog-salt'), // fixed salt
```

The PBKDF2 salt is hardcoded and identical across all users and sessions. Salt exists to prevent precomputation attacks; a fixed salt means all users with the same passphrase produce the same derived key. Combined with the fingerprint fallback (C1), this further weakens the encryption.

**Recommendation:** Generate a random salt per encryption, store it alongside the IV + ciphertext.

### C3. No authorization check -- any valid GitHub token grants full CMS access
**Files:** `src/components/admin/context/GitHubContext.tsx`, `src/components/admin/lib/github-api.ts`
**Category:** OWASP A01 (Broken Access Control)

`validateToken()` calls `octokit.rest.users.getAuthenticated()` and succeeds for *any* valid GitHub token, even one belonging to a stranger. There is no check that the authenticated user is the repository owner or an authorized collaborator. Anyone with a GitHub PAT can log in and, if they also have write scope on the repo, modify content.

**Recommendation:** After `validateToken()`, verify `data.login` against an allowlist (e.g., `["jiunbae"]`) or check that the user has push access to the specific repository via `octokit.rest.repos.getCollaboratorPermissionLevel()`.

### C4. XSS via unsanitized markdown rendering
**File:** `src/components/admin/components/MarkdownEditor.tsx`
**Category:** OWASP A03 (Injection)

`@uiw/react-md-editor` renders live preview of user-supplied markdown. By default it supports raw HTML in markdown. If an attacker can inject content (e.g., by loading a malicious draft or editing content that contains `<script>` tags, `<img onerror=...>`, or `javascript:` links), the preview will execute arbitrary JS in the admin session context -- which has access to the GitHub PAT.

**Recommendation:** Configure the MDEditor with `sanitize` or a custom `previewOptions.components` that strips dangerous HTML. Alternatively, use `rehype-sanitize` plugin.

---

## High Priority

### H1. Path traversal in content paths -- no validation that slug stays within content directories
**Files:** `src/components/admin/lib/content-paths.ts`, `src/components/admin/views/EditorView.tsx`
**Category:** OWASP A01 (Broken Access Control), OWASP A03 (Injection)

`generateContentPath()` directly interpolates `slug` and `date` into the file path:
```typescript
return `${dir}/${dateStr}-${slug}/index.md`;
```

`sanitizeSlug()` replaces non-alphanumeric chars with `-`, but does **not** strip `..` sequences or leading `/`. A crafted slug like `../../../etc/passwd` would become `------etc-passwd` after sanitization, which is safe. However, the slug field in the frontmatter form allows direct editing (line 89 of `FrontmatterForm.tsx`), and `handleSave` in `EditorView.tsx` only calls `sanitizeSlug()` on the slug used for path generation -- but the `path` variable for existing content (`path && !isNew ? path : ...`) is passed straight through without sanitization (line 135-136).

While the GitHub API itself limits what paths can be written, a malicious path could still create files outside the intended content directories via the API (e.g., overwriting `.github/workflows/` files to achieve RCE on the repository).

**Recommendation:** Validate that `targetPath` starts with one of the `CONTENT_DIRS` prefixes before sending to the API. Reject paths containing `..`.

### H2. No image file size or type validation on upload
**File:** `src/components/admin/components/ImageUploader.tsx`
**Category:** OWASP A04 (Insecure Design)

`processFile()` only checks `file.type.startsWith("image/")` (client-side MIME type, trivially spoofed). There is no file size limit. An attacker (or mistake) could upload a multi-GB file, causing the browser to OOM during base64 conversion or commit an enormous binary to the repository.

The filename is also used directly in `generateImagePath()` without sanitization -- filenames with spaces, special chars, or very long names could cause issues.

**Recommendation:** Add a file size limit (e.g., 5MB). Validate/sanitize the filename. Consider server-side validation if possible.

### H3. Stale SHA causes silent data loss on concurrent edits (race condition)
**Files:** `src/components/admin/views/EditorView.tsx`, `src/components/admin/hooks/useGitHub.ts`
**Category:** Data Integrity

When editing content, the SHA is fetched once on load and stored in component state. If two browser tabs edit the same file, or if a CI pipeline commits in between, the second save will fail with a 409 Conflict from the GitHub API -- but there is no retry logic or user guidance for conflict resolution. Worse, `saveContent` in `useGitHub.ts` updates the cache optimistically (line 132), so after a failed save the cache may be stale.

The `saveContent` function catches the error and re-throws it, but `EditorView` only shows a generic error message. The user has no way to know their content was not saved or to merge changes.

**Recommendation:** On 409 Conflict, fetch the latest version, show a diff, and let the user resolve. At minimum, clearly communicate that the save failed due to a conflict and suggest refreshing.

### H4. Token not cleared from React state on logout race condition
**File:** `src/components/admin/context/GitHubContext.tsx`
**Category:** OWASP A07 (Identification and Authentication Failures)

`logout()` sets `octokit` to `null` and calls `removeGitHubToken()`, but the Octokit instance may still be referenced by in-flight API calls (closures in `useGitHubAPI`). If a save operation is in progress during logout, it could complete using the old token after the user believes they have logged out.

**Recommendation:** Use an AbortController or token generation counter to invalidate in-flight requests on logout.

---

## Medium Priority

### M1. `any` types used extensively, undermining TypeScript safety
**Files:** Multiple
**Category:** Type Safety

- `src/components/admin/hooks/useAutoSave.ts:7` -- `frontmatter: any`
- `src/components/admin/views/EditorView.tsx:35` -- `[key: string]: any` index signature on Frontmatter
- `src/components/admin/views/EditorView.tsx:138` -- `fmToSerialize: Record<string, any>`
- `src/components/admin/lib/frontmatter.ts` -- `Record<string, any>` throughout, `parseValue` returns `any`
- `src/components/admin/lib/github-api.ts:3` -- `(import.meta as any).env`
- `src/components/admin/views/EditorView.tsx:183` -- Draft cast `as Frontmatter`

The index signature `[key: string]: any` on `Frontmatter` means any typo compiles silently and the type provides no protection.

**Recommendation:** Define a strict `Frontmatter` type without the index signature. Use a separate `Record<string, unknown>` type for the raw parsed YAML. Create proper Vite env type declarations to avoid `(import.meta as any)`.

### M2. `import.meta` cast to `any` hides env type errors
**File:** `src/components/admin/lib/github-api.ts:3-5`
**Category:** Type Safety

```typescript
export const OWNER = (import.meta as any).env?.PUBLIC_GITHUB_OWNER ?? "jiunbae";
```

Casting to `any` suppresses all type checking. Vite provides `ImportMeta` type augmentation for this purpose.

**Recommendation:** Add a `src/env.d.ts` with proper `ImportMetaEnv` interface declaration.

### M3. Custom YAML parser is fragile and incomplete
**File:** `src/components/admin/lib/frontmatter.ts`
**Category:** Architecture (Robustness)

The hand-rolled YAML parser does not handle:
- Nested objects
- Multi-line strings (block scalars `|`, `>`)
- Anchors/aliases
- Flow mappings `{key: value}`
- Comments after values on the same line (e.g., `title: Hello # comment`)
- Escaped characters beyond `\"` and `\'`
- YAML 1.2 numeric formats (octal `0o`, hex `0x`)

This is a CMS that will parse arbitrary existing frontmatter. If any existing post uses features not supported, it will silently corrupt or drop data on round-trip.

**Recommendation:** Use an established YAML library (`yaml`, `js-yaml`) that handles the full spec. The bundle-size cost is minimal for an admin page.

### M4. `useGitHubAPI` hook creates a new instance per component -- no shared state
**File:** `src/components/admin/hooks/useGitHub.ts`
**Category:** Architecture (Separation of Concerns)

Each component calling `useGitHubAPI()` gets its own `loading`, `error`, `tree`, and `contentCacheRef`. This means:
- `ContentItem` (rendered N times in the list) each creates a separate cache ref
- Loading/error state in one component is invisible to others
- The `loading` boolean is shared within a single hook instance but set by multiple async operations, causing state thrashing

**Recommendation:** Lift the API state into a React Context (similar to `GitHubContext`) or use a data-fetching library (React Query / SWR) with proper cache sharing.

### M5. `ContentItem` fetches file content eagerly for every item in the list
**File:** `src/components/admin/components/ContentItem.tsx`
**Category:** Performance / API Design

The list view renders a `ContentItem` for every file, and each one triggers `fetchContent()` to parse frontmatter. For a repository with 50+ content files, this fires 50+ GitHub API requests simultaneously. The GitHub API has rate limits (5000/hour for authenticated, lower for fine-grained tokens).

**Recommendation:** Implement pagination or virtual scrolling. Consider extracting frontmatter from the tree API response if possible, or fetch content lazily (on hover/scroll into view).

### M6. Browser fingerprint is unstable -- token becomes unrecoverable
**File:** `src/utils/crypto.ts:88-103`
**Category:** OWASP A02 (Cryptographic Failures)

The fingerprint includes `navigator.userAgent` (changes on browser update), `screen.width + 'x' + screen.height` (changes on monitor swap or display scaling change), and `navigator.language` (can change). If any of these change, the stored encrypted token becomes permanently unrecoverable without any error message explaining why.

**Recommendation:** If keeping encryption, do not use volatile browser properties as key material. Require the passphrase and store it in `sessionStorage` (accepting it is lost on tab close).

---

## Low Priority / Suggestions

### L1. Duplicate Frontmatter interface definitions
**Files:** `src/components/admin/views/EditorView.tsx:25-36`, `src/components/admin/components/FrontmatterForm.tsx:4-13`
**Category:** Architecture (DRY)

Two slightly different `Frontmatter` interfaces exist. The `EditorView` version has `rating` and `[key: string]: any`; the `FrontmatterForm` version does not. This invites drift.

**Recommendation:** Define a single canonical `Frontmatter` type in a shared types file.

### L2. `fetchTree` in `useEffect` missing from dependency array
**File:** `src/components/admin/views/ListView.tsx:22`
**Category:** React Correctness

```typescript
useEffect(() => { fetchTree(); }, []);
```

`fetchTree` is omitted from the dependency array. While this is intentional (fire once on mount), it triggers the `react-hooks/exhaustive-deps` lint rule and could cause subtle bugs if `fetchTree` identity changes.

**Recommendation:** Either add `fetchTree` to the dependency array (ensuring it is stable via `useCallback` with stable deps) or use an explicit `// eslint-disable-next-line` comment documenting the intent.

### L3. Error messages leak implementation details
**Files:** Multiple (`useGitHub.ts`, `GitHubContext.tsx`, `EditorView.tsx`)
**Category:** OWASP A09 (Security Logging and Monitoring Failures)

Raw error messages from the GitHub API (which can include repository names, auth scopes, and rate limit details) are rendered directly in the UI via `err.message`.

**Recommendation:** Map known error codes to user-friendly messages. Log the raw error to the console for debugging but show a sanitized message to the user.

### L4. Console.error calls in production code
**Files:** `src/utils/crypto.ts:155,188`, `src/utils/storage.ts:73,99`, `src/components/admin/components/ImageUploader.tsx:42`
**Category:** OWASP A09 (Security Logging and Monitoring Failures)

`console.error('Encryption failed:', error)` and similar calls will print potentially sensitive data (encrypted tokens, error details) to the browser console in production.

**Recommendation:** Use a logging abstraction that can be configured per environment, or strip console calls in production builds.

### L5. `encodeBase64` / `decodeBase64` use deprecated `escape()`/`unescape()`
**File:** `src/components/admin/lib/github-api.ts:152-165`
**Category:** Code Quality

```typescript
return btoa(unescape(encodeURIComponent(str)));
```

`escape()` and `unescape()` are deprecated. While the pattern `btoa(unescape(encodeURIComponent()))` is a well-known hack for Unicode in base64, it is fragile and confusing.

**Recommendation:** Use `TextEncoder`/`TextDecoder` with manual base64 conversion, or a small utility function.

### L6. Draft data is never pruned or size-limited
**File:** `src/utils/storage.ts`
**Category:** Robustness

Drafts accumulate indefinitely in localStorage with no maximum count or total size limit. Auto-save runs every 5 seconds. Over time this could fill localStorage (typically ~5MB limit), causing silent failures.

**Recommendation:** Limit draft count (e.g., keep last 20). Add a total size check. Prune old drafts on app startup.

### L7. `uploadImage` does not check for existing file -- overwrites silently
**File:** `src/components/admin/lib/github-api.ts:132-150`
**Category:** Data Integrity

`uploadImage` calls `createOrUpdateFileContents` without providing a `sha`, so it will create a new file. But if a file with the same name already exists, the API will return a 422 error (SHA required for update). There is no handling for this case.

**Recommendation:** Check if the file exists first, or handle the 422 gracefully by fetching the existing SHA and updating.

### L8. `admin.astro` is publicly accessible -- no route-level protection
**File:** `src/pages/admin.astro`
**Category:** Defense in Depth

The admin page is served to any visitor. While the React app requires authentication, the page itself (including all admin JS/CSS bundles) is downloaded by anyone. This exposes the admin UI code to reconnaissance.

**Recommendation:** This is acceptable for a static site with client-side auth, but consider adding a `robots.txt` disallow for `/admin` and/or a `<meta name="robots" content="noindex">` tag.

---

## Statistics
- Total findings: 18
- Critical: 4, High: 4, Medium: 6, Low: 8
