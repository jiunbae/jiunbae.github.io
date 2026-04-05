# Performance & Code Quality Review (R02)

Scope: `src/components/admin/`, `src/pages/admin.astro`, `src/utils/crypto.ts`, `src/utils/storage.ts`

---

## Critical Findings (must fix)

### C1. Context value object re-created every render, causing full subtree re-renders
**File:** `src/components/admin/context/GitHubContext.tsx:111-119`

The `value` object passed to `<GitHubContext.Provider>` is constructed inline on every render. Because it is a new object reference each time, every consumer of `useGitHub()` will re-render on every GitHubProvider render, even if no individual field changed. This cascades to ListView, EditorView, and every hook that calls `useGitHub()` or `useGitHubAPI()`.

**Fix:** Wrap `value` in `useMemo` keyed on its constituent fields (`octokit`, `user`, `isAuthenticated`, `isLoading`, `error`, `login`, `logout`).

### C2. `useGitHubAPI()` is a plain hook, not shared state -- each consumer gets independent instances
**File:** `src/components/admin/hooks/useGitHub.ts`

`useGitHubAPI()` contains its own `useState` for `loading`, `error`, and `tree`. Every component that calls `useGitHubAPI()` gets a completely independent copy of these states. This means:
- `ListView` calls `useGitHubAPI()` and fetches the tree.
- `EditorView` calls `useGitHubAPI()` independently and gets its own `tree: null`, its own `loading: false`, its own `contentCacheRef` -- the cache from ListView is not shared.
- `ImageUploader` calls `useGitHubAPI()` again -- yet another independent instance.

The `contentCacheRef` in particular is per-hook-instance, so there is zero caching benefit across components. If two components fetch the same file, two API calls are made.

**Fix:** Either lift `useGitHubAPI` state into a context provider (like `GitHubContext`), or create a dedicated `GitHubAPIProvider` that holds the shared cache and loading state.

### C3. `useAutoSave` creates a separate `useDrafts` instance, causing duplicate localStorage reads
**File:** `src/components/admin/hooks/useAutoSave.ts:22`

`useAutoSave` calls `useDrafts()` internally, while `DraftManager` also calls `useDrafts()`. These are independent hook instances with independent `drafts` state arrays. Every 5 seconds when auto-save fires, `useAutoSave`'s `saveDraft` calls `storageSaveDraft` then `getDrafts()` (which reads and parses localStorage). But the `DraftManager`'s `drafts` state never updates -- it only updates on explicit `refreshDrafts()`. This is both wasteful and semantically confusing.

**Fix:** Accept `saveDraft` as a parameter to `useAutoSave` instead of instantiating `useDrafts` internally, or consolidate draft state into a shared context.

---

## High Priority

### H1. `useAutoSave` triggers on every frontmatter reference change -- causes save on every keystroke after debounce
**File:** `src/components/admin/hooks/useAutoSave.ts:74`

The effect dependency list includes `data.frontmatter`. In `EditorView`, `frontmatter` is a new object reference on every `setFrontmatter` call (every keystroke in any form field). The debounce timer resets correctly, but the effect itself runs on every keystroke, creating and clearing timeouts at high frequency. More critically, `data.frontmatter` as a dependency uses reference equality, so the timer resets even if the actual frontmatter content hasn't changed.

**Fix:** Either use a deep-comparison custom hook, or track frontmatter via a serialized/stable key (e.g., `JSON.stringify(data.frontmatter)`).

### H2. `@octokit/rest` and `@uiw/react-md-editor` are large dependencies loaded for the admin page
**Files:** `package.json`, `src/pages/admin.astro`

`@octokit/rest` pulls in the entire Octokit surface area (dozens of endpoint methods). `@uiw/react-md-editor` is a full-featured markdown editor with preview. Both are loaded via `client:only="react"` which means they are only client-side, but they are still part of the main JS bundle for anyone who visits `/admin`.

Since this is an admin-only page, this is acceptable for now, but:
- `@octokit/rest` could be replaced with `@octokit/request` (much smaller) since only 5 endpoints are used.
- The MDEditor could be lazy-loaded (`React.lazy`) since it is only needed in the editor view, not the list view.

### H3. `ListView` re-computes filtered+sorted entries on every render without memoization
**File:** `src/components/admin/views/ListView.tsx:25-36`

`filteredEntries` calls `parseContentPath` three times per entry (twice in the filter, once in the sort comparator). For a repository with N content files, this is O(N * 3) regex operations on every render. The result should be memoized with `useMemo`.

### H4. Missing `fetchTree` in `useEffect` dependency array
**File:** `src/components/admin/views/ListView.tsx:21-23`

```tsx
useEffect(() => {
  fetchTree();
}, []);
```

`fetchTree` is omitted from the dependency array. While this "works" because `fetchTree` is stable due to `useCallback`, the ESLint `react-hooks/exhaustive-deps` rule would flag this. More importantly, if `octokit` changes (e.g., re-authentication), `fetchTree` would get a new reference but the effect would not re-run, leaving the list showing stale data.

### H5. `DEFAULT_FRONTMATTER` uses `new Date()` at module load time
**File:** `src/components/admin/views/EditorView.tsx:38-47`

`DEFAULT_FRONTMATTER` is a module-level constant whose `date` field is set to `new Date().toISOString().slice(0, 10)`. This is evaluated once when the module is first imported. If the user keeps the tab open past midnight, new posts will have yesterday's date. Additionally, the object is spread into state (`{ ...DEFAULT_FRONTMATTER }`) but this is a shallow copy -- the `tags: []` array is shared across all instances. Not a current bug since arrays are replaced not mutated, but fragile.

**Fix:** Make it a factory function: `const createDefaultFrontmatter = () => ({ ... })`.

---

## Medium Priority

### M1. `FrontmatterForm.update` callback closes over current `frontmatter` -- re-created every keystroke
**File:** `src/components/admin/components/FrontmatterForm.tsx:26-31`

The `update` callback has `[frontmatter, onChange]` as dependencies. Since `frontmatter` is a new object on every change, `update` is re-created on every keystroke. All child event handlers that depend on `update` are also re-created. This triggers unnecessary re-renders for all form inputs.

**Fix:** Use a functional updater pattern or accept a callback-based onChange to avoid depending on the current frontmatter value.

### M2. Unhandled promise in `ImageUploader.handleDrop`
**File:** `src/components/admin/components/ImageUploader.tsx:62-74`

`processFile(files[0])` returns a Promise, but the result is not awaited or caught in `handleDrop`. If `processFile` throws, this becomes an unhandled promise rejection. The same issue exists conceptually in `handleFileSelect` (line 79).

**Fix:** Add `.catch()` or make the handlers async.

### M3. `getBrowserFingerprint` includes volatile components
**File:** `src/utils/crypto.ts:88-103`

The fingerprint includes `navigator.userAgent`. Browser updates change the UA string, which changes the derived encryption key, which silently makes previously-encrypted tokens undecryptable. The user would need to re-authenticate with no explanation why. `screen.width + 'x' + screen.height` also changes if the user connects an external monitor.

This is a design issue rather than a pure code quality issue, but it directly affects UX: users will randomly get logged out after browser updates.

### M4. No error boundary around the admin app
**File:** `src/components/admin/AdminApp.tsx`

There is no React error boundary. If any component throws during render (e.g., corrupt localStorage data, unexpected API response shape), the entire admin UI crashes to a white screen with no recovery path.

**Fix:** Add an error boundary component wrapping `AdminRouter`.

### M5. `contentCache` exposed as mutable Map reference
**File:** `src/components/admin/hooks/useGitHub.ts:199`

`contentCache: contentCacheRef.current` exposes the raw `Map` object. Any consumer can mutate it directly, bypassing the hook's state management. This also means mutations won't trigger re-renders.

### M6. Shared `loading` state across all API operations in `useGitHubAPI`
**File:** `src/components/admin/hooks/useGitHub.ts:47`

A single `loading` boolean is used for `fetchTree`, `fetchContent`, `saveContent`, `deleteContent`, and `uploadImage`. If a tree fetch is in progress and the user triggers a save, `loading` flips to `true` then `false` when save completes, even though the tree fetch is still in progress. This creates incorrect UI states.

**Fix:** Use separate loading states per operation, or use a counter/set of in-flight operations.

### M7. YAML parser is hand-rolled and incomplete
**File:** `src/components/admin/lib/frontmatter.ts:88-119`

The custom YAML parser does not handle: nested objects, multi-line strings (block scalars `|` / `>`), anchors/aliases, flow mappings, or escaped characters in keys. If any existing frontmatter uses these features, the parser will silently drop or misparse data.

This is a deliberate tradeoff (avoiding a YAML library dependency), but it should be documented with the known limitations, and ideally validated with tests.

### M8. `encodeBase64`/`decodeBase64` use deprecated `escape`/`unescape`
**File:** `src/components/admin/lib/github-api.ts:152-165`

`unescape(encodeURIComponent(str))` and `decodeURIComponent(escape(...))` use the deprecated `escape`/`unescape` functions. While they work for UTF-8 encoding, they are removed from the spec and may be dropped by future engines.

**Fix:** Use `TextEncoder`/`TextDecoder` with `Uint8Array` for proper UTF-8 base64 encoding, consistent with what `crypto.ts` already does.

---

## Low Priority / Suggestions

### L1. Duplicate `Frontmatter` interface definition
**Files:** `src/components/admin/views/EditorView.tsx:25-36`, `src/components/admin/components/FrontmatterForm.tsx:4-13`

The `Frontmatter` interface is defined independently in both files. They are mostly identical but EditorView's version has extra fields (`rating`, index signature). Extract a shared type.

### L2. `DraftManager` wrapper callbacks add no value
**File:** `src/components/admin/components/DraftManager.tsx:31-40`

`handleDelete` simply forwards to `deleteDraft`, and `handleDeleteAll` simply forwards to `deleteAllDrafts`. These `useCallback` wrappers are unnecessary indirection -- pass the original functions directly.

### L3. `ConfirmDialog` uses inline styles
**File:** `src/components/admin/components/ConfirmDialog.tsx:30,34,35`

Uses `style={{ maxWidth: 400 }}` and `style={{ display: "flex", ... }}` instead of CSS classes. Inconsistent with the rest of the codebase which uses class-based styling.

### L4. `ContentItem` accepts `meta` and `loading` props that are never passed
**File:** `src/components/admin/components/ContentItem.tsx:6-14`

The `meta` and `loading` props are defined in the interface but `ListView` never passes them. This is dead code / premature abstraction. The loading branch and meta-based rendering are unreachable.

### L5. `parseContentPath` called redundantly in `ListView`
**File:** `src/components/admin/views/ListView.tsx:30,33-34`

`parseContentPath` is called once in the filter (line 30), then again twice in the sort comparator (lines 33-34) for the same paths. Cache the parsed results.

### L6. `AdminRouter` default case duplicates `list` case
**File:** `src/components/admin/AdminApp.tsx:60-61`

The `default` case in the switch statement is identical to the `list` case. Since `View` is a discriminated union, the default is unreachable. Remove it or make it an explicit exhaustiveness check.

### L7. Magic number for auto-save delay
**File:** `src/components/admin/views/EditorView.tsx:76`

Auto-save is always enabled (`true`) with the default 5000ms delay. Consider making this configurable or at least extracting it as a named constant.

### L8. `PBKDF2` with 100k iterations on every encrypt/decrypt
**File:** `src/utils/crypto.ts:70-81`

Key derivation runs 100,000 PBKDF2 iterations on every `encrypt()` and `decrypt()` call. For auto-save (every 5s), this is fine. But if multiple encryptions happen in quick succession, it could cause noticeable UI jank. Consider caching the derived key for the session lifetime.

---

## Statistics
- Total findings: 20
- Critical: 3, High: 5, Medium: 8, Low: 8
