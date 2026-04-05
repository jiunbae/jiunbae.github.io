# Code Review Summary (R02) — Admin CMS PR #40

## Reviewers
| Agent | Perspective | Findings |
|-------|-----------|----------|
| Gemini Code Assist | Inline PR review | 8 |
| Claude (Architecture) | Security + architecture | 18 |
| Claude (Performance) | Performance + code quality | 20 |

## Applied Fixes

### Critical (fixed)
- **C3/Auth**: User allowlist in `validateToken()` — only `jiunbae` can access CMS
- **C4/XSS**: Added `rehype-sanitize` to MarkdownEditor preview
- **C1/Perf**: `useMemo` for GitHubContext value to prevent subtree re-renders

### High (fixed)
- **H1/Path traversal**: Validate `targetPath` within `CONTENT_DIRS` before API call
- **H2/Image size**: 5MB file size limit + filename sanitization
- **H3/Conflict**: 409 Conflict detection with user-friendly message + cache invalidation
- **H5/Date**: `DEFAULT_FRONTMATTER` → `createDefaultFrontmatter()` factory

### Medium (fixed via Gemini)
- `parseContentPath()` instead of `path.includes()`
- `fetchContent` in ContentItem useEffect deps
- `message_` → `errorMessage`
- ENV vars for OWNER/REPO/BRANCH
- `currentDraftId` passed to DraftManager
- Inline styles → SCSS classes
- `noindex` meta tag on admin page

### Other (fixed)
- Error boundary wrapping AdminRouter

## Deferred (acceptable trade-offs for v1)

### Security
- **C1/C2 (crypto)**: Browser fingerprint fallback + fixed salt — mitigated by passphrase option. Full fix requires OAuth flow (out of scope).
- **H4 (logout race)**: In-flight requests may use stale token — low risk for single-user CMS.

### Performance
- **C2 (shared state)**: `useGitHubAPI` independent per-component — acceptable for small content sets. Lift to context in v2 if needed.
- **C3 (useDrafts)**: Duplicate instances in useAutoSave + DraftManager — low impact (localStorage reads are fast).
- **H1 (autosave deps)**: Timer resets per keystroke — debounce works correctly, just unnecessary effect runs.
- **H3 (memoize filter)**: `filteredEntries` recomputed on render — negligible for <100 items.

### Code Quality
- **M3/M7 (YAML parser)**: Custom parser handles blog's frontmatter format. Full YAML lib for v2.
- **L1 (duplicate types)**: Extract shared Frontmatter type in v2.
- **M8 (deprecated escape/unescape)**: Already replaced with TextEncoder/TextDecoder.

## Statistics
- Total unique findings: ~38 (deduplicated across reviewers)
- Fixed: 14
- Deferred: ~12
- Not applicable / already resolved: ~12
