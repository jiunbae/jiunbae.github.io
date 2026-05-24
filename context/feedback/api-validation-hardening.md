# API Validation & Hardening — Research Brief

Target: `jiun-api` (Hono + Bun + MongoDB) consumed by Astro static site +
`CommentSection.tsx` (React text nodes) + satori-rendered OG images.

Sources: OWASP ASVS V5 (Validation, Sanitization, Encoding), OWASP NoSQL
Injection Prevention CS, OWASP XSS Prevention CS, Hono validator docs, Zod
docs, OWASP Mass Assignment CS, Unicode UAX #31 / UAX #9 (bidi).

---

## A. `postSlug` mass-assignment / spam vector

**Best practice (OWASP Mass Assignment CS):** never trust client-supplied
identifiers that scope or bind a resource. Either bind from server context, or
prove the client received the identifier from a trusted source. ASVS V5.1.4
requires that "untrusted data is validated against a defined schema, allowlist,
or context-bound credential."

**Current code:** `src/routes/comments.ts:106` — `postSlug: body.postSlug`
is taken from JSON body and only length-checked at
`src/services/comment.service.ts:230-235`. An attacker can POST to any slug
including unpublished future posts.

**Recommendation — combined defense (regex + HMAC binding):**

1. **Format allowlist (necessary, not sufficient).**
   `^[a-z0-9][a-z0-9-]{0,199}$` — strict kebab-case. Rejects path traversal,
   operator chars, NUL, whitespace.

2. **HMAC slug-sig pattern (RECOMMENDED).** Astro renders each page with a
   short-lived signature; React component sends `(slug, postType, sig)`; API
   verifies HMAC. No DB sync required, fully static-compatible.

   ```ts
   // src/lib/slugSig.ts (API + Astro share this file via package)
   import crypto from 'node:crypto';
   const SECRET = process.env.SLUG_SIGNING_SECRET!; // 32+ bytes random
   export function generateSlugSig(slug: string, postType: string): string {
     return crypto.createHmac('sha256', SECRET)
       .update(`${postType}:${slug}`)
       .digest('base64url')
       .slice(0, 22); // 132-bit truncation, plenty for spam-gate
   }
   export function verifySlugSig(slug: string, postType: string, sig: string): boolean {
     const expected = generateSlugSig(slug, postType);
     if (sig.length !== expected.length) return false;
     return crypto.timingSafeEqual(Buffer.from(sig), Buffer.from(expected));
   }
   ```

   Astro page emits:

   ```astro
   ---
   import { generateSlugSig } from '~/lib/slugSig';
   const sig = generateSlugSig(slug, 'posts');
   ---
   <CommentSection client:visible postSlug={slug} postType="posts" slugSig={sig} />
   ```

   API verifies (after Zod parse):

   ```ts
   if (!verifySlugSig(body.postSlug, body.postType, body.slugSig)) {
     throw new HttpError(403, 'invalid slug signature');
   }
   ```

3. **Why not an existence check?** Build-time sync (POST list of slugs +
   HMAC) works but couples build pipeline to API uptime and breaks for drafts.
   HMAC is stateless and equally strong against spam.

4. **Legacy migration:** ship `verifySlugSig` in **warn mode** for 2 weeks
   — log invalid sigs to a metric, accept anyway. Then flip a feature flag
   (`REQUIRE_SLUG_SIG=true`) once frontend deploy is stable. Existing comments
   in DB are unaffected (they were created via the un-signed flow).

`SLUG_SIGNING_SECRET` must live in both the Astro build env and the API env;
it is not a session secret — rotating it invalidates all sigs at once.

---

## B. Server-side input validation across the API

**Best practice (ASVS V5.1):** declarative schema validation at the edge,
fail-closed, structured error responses. Hono docs explicitly recommend the
`@hono/zod-validator` middleware for typed boundaries.

**Current code:**
- `src/routes/comments.ts:102` — `await c.req.json()` returns `any`, fields
  used with `.trim()` would throw `TypeError` on non-string.
- `src/routes/novaPouch.ts:87-103` — only `typeof body.story === 'string'`.
- `src/routes/game.ts:23-33` — `body.stats` passed through with no shape check.
- `src/routes/auth.ts:284-292` — only checks `typeof code === 'string'`.

**Library choice — recommend Zod.**

| | Zod | Valibot | Hono built-in |
|---|---|---|---|
| Bundle (gzip) | ~14 KB | ~3 KB | 0 |
| Hono integration | `@hono/zod-validator` (1st-party) | `@hono/valibot-validator` | manual |
| Type inference | Excellent | Excellent | None |
| Ecosystem (refinements, transforms) | Mature | Newer | n/a |

Bun starts cold-fast enough that 14 KB is negligible at runtime. Zod has the
best Hono+TS dev ergonomics and the team already knows it (per ecosystem
defaults). **Pick Zod.** Reconsider Valibot only if you target Cloudflare
Workers cold-starts later.

**Migration pattern:**

```ts
// src/schemas/comment.ts
import { z } from 'zod';
export const PostType = z.enum(['posts', 'notes', 'reviews']);
export const Slug = z.string().regex(/^[a-z0-9][a-z0-9-]{0,199}$/);
export const ObjectIdStr = z.string().regex(/^[a-f0-9]{24}$/);

export const CreateCommentBody = z.object({
  postSlug: Slug,
  postType: PostType.default('posts'),
  parentId: ObjectIdStr.optional(),
  anonName: z.string().min(1).max(50).optional(),
  content: z.string().min(1).max(3000),
  slugSig: z.string().length(22), // stage 2: required
}).strict(); // .strict() rejects extra fields → defense vs mass-assignment
```

```ts
// src/routes/comments.ts
import { zValidator } from '@hono/zod-validator';
commentRoutes.post('/',
  optionalAuth,
  zValidator('json', CreateCommentBody, (result, c) => {
    if (!result.success) {
      return c.json({ error: 'validation_error', issues: result.error.issues }, 400);
    }
  }),
  async (c) => {
    const body = c.req.valid('json');     // typed
    // ...
  }
);
```

Apply per resource:
- `src/schemas/comment.ts` — list/count/counts/create/update bodies & queries.
- `src/schemas/auth.ts` — `provider` enum, `code` (32-byte hex), `redirect_uri`
  url+allowlist (already exists).
- `src/schemas/novaPouch.ts` — `story` (max 5000), `tokens` (3 token IDs from
  enum), `date` (YYYY-MM-DD ISO).
- `src/schemas/game.ts` — `score` int range, `ending` enum, `stats` shape.

Use `z.coerce.number().int().positive().max(N)` to replace `parsePositiveInt`.

---

## C. NoSQL operator injection

**Best practice (OWASP NoSQL Injection CS):** treat any field placed into a
filter/update as typed input. Reject objects where strings are expected. Never
spread request bodies into filters or `$set`.

**Current code:** not exploitable today because Mongo driver receives
already-stringified values (e.g. `comments.findOne({ _id: objectId })`), BUT:
- `src/routes/comments.ts:102` passes raw `body.parentId` to `parseObjectId`
  — which would throw on non-string, but a `{ "$regex": "^..." }` object
  passed deeper could be dangerous if any future code uses it pre-validation.
- `src/services/novaPouch.service.ts:184` builds `recordToInsert` from
  `input.anonName?.trim().slice(0,50)` — safe because `.trim()` throws on
  non-string, but Zod makes the guarantee explicit.

**Pattern — safe update:**

```ts
// Bad (hypothetical future code):
await comments.updateOne({ _id: objectId }, { $set: body });

// Good — explicit allowlist mapping, types come from Zod:
const update: Partial<CommentDocument> = {
  content: body.content,
  updatedAt: new Date(),
};
await comments.updateOne({ _id: objectId, userId: userObjectId }, { $set: update });
```

Also note: **always include `userId` in the filter** for owner-scoped updates
(not just a separate `findOne` check) — defense in depth against TOCTOU.
`updateComment` currently does `findOne` then `updateOne` (lines 303 + 322).

---

## D. XSS strategy: sanitize-on-write vs escape-on-render

**Best practice (OWASP XSS Prevention CS Rules #1, #7):** escape at the
output context, not at storage. Different contexts (HTML body, attribute, JS,
URL, CSS) need different encodings — pre-escaping locks you into one context
and corrupts the others.

**Current bug:** `sanitizeContent` (`comment.service.ts:19-26`) HTML-escapes 5
chars at write. React then auto-escapes on render
(`CommentSection.tsx:524` — `<p>{comment.content}</p>`). Result: user typing
`<script>alert(1)</script>` sees `&lt;script&gt;alert(1)&lt;/script&gt;`
literally in the DOM. Korean comments containing `<`, `>`, `&`, `"`, `'` are
also corrupted.

**Recommendation:**

1. **Drop `sanitizeContent` from write path.** Store raw content after only:
   - control-char strip (U+0000-U+0008, U+000B-U+000C, U+000E-U+001F, U+007F)
   - normalize CRLF → LF
   - trim trailing whitespace
   - cap length

   ```ts
   export function normalizeContent(s: string): string {
     return s
       .replace(/\r\n/g, '\n')
       .replace(/[ --]/g, '')
       .trim();
   }
   ```

2. **Trust the renderer.** React/Astro text interpolation auto-escapes.
   Document this in the model: `CommentDocument.content` is "untrusted text;
   renderers MUST escape."

3. **Satori (OG image) is safe-by-default** for text children — satori turns
   strings into SVG `<text>` nodes, which are not parsed as HTML.
   `novaPouch.ts:406` passes `storyTeaser` as a `children` string property —
   confirmed safe. Risk only emerges if a future change passes user text into
   an `innerHTML` or sets an SVG `href` attribute.

4. **Dual-field hedge (optional).** If you anticipate future renderers that
   may use `dangerouslySetInnerHTML`, the API can return both:
   ```json
   { "content": "raw text", "contentSafe": "html-escaped text" }
   ```
   But this doubles payload — only add when a concrete consumer needs it.

5. **`anonName` hardening** (`comment.service.ts:260` —
   `input.anonName?.trim().slice(0, 50)`):
   - Strip control chars (as above)
   - Strip zero-width: U+200B-U+200F, U+202A-U+202E (LRE/RLE/PDF/LRO/RLO
     **bidi overrides** — used in homograph attacks per CVE-2021-42574
     "Trojan Source"), U+2066-U+2069, U+FEFF
   - Allow Unicode letters/marks/numbers/punctuation/space:
     `/^[\p{L}\p{M}\p{N}\p{P}\p{Zs}]+$/u` after stripping
   - Cap to 50 grapheme clusters (not bytes/UTF-16 units) using
     `Intl.Segmenter`

   ```ts
   export function sanitizeAnonName(raw: string): string {
     const stripped = raw
       .normalize('NFC')
       .replace(/[ -​-‏‪-‮⁦-⁩﻿]/g, '')
       .trim();
     if (!/^[\p{L}\p{M}\p{N}\p{P}\p{Zs}]+$/u.test(stripped)) {
       throw new HttpError(400, 'Invalid characters in anonName');
     }
     const segs = [...new Intl.Segmenter().segment(stripped)].slice(0, 50);
     return segs.map(s => s.segment).join('');
   }
   ```

---

## E. Markdown / formatting in comments

`CommentSection.tsx:524` renders `<p style={{whiteSpace:'pre-wrap'}}>{content}</p>` —
**no markdown**. Linebreaks and spaces are preserved via CSS; URLs are NOT
auto-linkified.

**Policy to document:** comments are plain text. If markdown is added later,
use `react-markdown` with `disallowedElements={['script','iframe','style','link','meta']}` and `skipHtml`, OR `markdown-it` with `html: false, linkify: true`.
Either way the renderer must run client-side; do **not** pre-render markdown
on the server (re-introduces the storage-vs-render coupling).

---

## F. NovaPouch HTML render audit

`novaPouch.ts:137-238` builds two HTML pages. `escapeHtml` (lines 183-190)
escapes `& < > " '` — the canonical 5 chars. **Verdict: correct for HTML body
and attribute contexts** as currently used.

Notes:
- `<meta http-equiv="refresh" content="0;url=${escapeHtml(redirectUrl)}">` —
  the `url=` is inside `content="..."` so attribute escaping suffices. But
  `redirectUrl` derives from `config.frontendUrl + '/?r=' + id` where `id`
  comes from `c.req.param('id')` — not validated as an ObjectId hex string at
  this point. Recommend: validate `id` against `^[a-f0-9]{24}$` before
  building URL.
- `imageUrl` is hardcoded `https://api.jiun.dev/...` — fine.
- The `<a href="${escapeHtml(redirectUrl)}">` is safe for syntactic injection
  but does **not** prevent a `javascript:` scheme if `frontendUrl` were ever
  attacker-controlled (it isn't — env-only). Still, add a scheme allowlist in
  a helper.
- **CSP** (out of scope here) should be set on these responses too.

---

## G. Comment edit/delete authorization

`comment.service.ts:310` and `:366` both check
`!comment.userId || !comment.userId.equals(userObjectId)` before mutation.
**No admin-override bypass exists** — grep for `isAdmin`, `admin`, `role`
across `src/` returned no policy hooks. Anon comments (`userId === null`) are
**uneditable and undeletable by anyone**, which is correct behavior given the
threat model but worth documenting as a product decision (no recovery path
for anon spam beyond moderator DB action).

Defense-in-depth recommendation: also scope the `updateOne`/`updateOne`
filter by `userId` (see §C) so the auth check is enforced at the DB layer,
not just in JS.

---

## H. File upload / OG image generation

`/nova-pouch/records/:id/og-image.png` (lines 262-468) generates a PNG via
satori + resvg. User text enters via `story` field, capped at 5000 chars,
then truncated to 150 (`storyTeaser`) for the image.

**Risks & mitigations:**

1. **Render-time DOS.** Satori is CPU-bound (~50-200 ms for 1200×630).
   Currently no rate limit on this route — add per-IP rate limit (existing
   `rateLimiter.ts` middleware).
2. **Font fallback.** `loadFont()` fetches Noto Sans KR at runtime from
   Google Fonts and caches in-process. If glyphs are missing (e.g.
   emoji, CJK extension B, or RTL chars), satori falls back to **rendering
   the codepoint as `?` or skipping** — no crash, but visible breakage. Note
   that `tokens.red.emoji` etc. **are pre-defined server-side** so emoji
   coverage is bounded; only `story` is user-provided.
3. **Length already bounded** (5000 store, 150 render) — good.
4. **Cache header** `public, max-age=86400` — fine. Consider adding an
   `ETag` keyed on `record._id` so cache buster works on edit (records are
   currently immutable per the codebase, so unnecessary today).
5. **No image upload** in the codebase — confirmed.

---

## Proposed Zod schema bundle (`src/schemas/comment.ts`)

```ts
import { z } from 'zod';

export const PostType = z.enum(['posts', 'notes', 'reviews']);
export const Slug = z.string().regex(/^[a-z0-9][a-z0-9-]{0,199}$/, 'invalid slug');
export const ObjectIdStr = z.string().regex(/^[a-f0-9]{24}$/, 'invalid id');
export const SlugSig = z.string().length(22).regex(/^[A-Za-z0-9_-]+$/);

// stage 1: slugSig optional (warn mode); stage 2: required
export const CreateCommentBody = z.object({
  postSlug: Slug,
  postType: PostType.default('posts'),
  parentId: ObjectIdStr.optional(),
  anonName: z.string().min(1).max(50).optional(),
  content: z.string().min(1).max(3000),
  slugSig: SlugSig.optional(), // stage 2: remove .optional()
}).strict();

export const UpdateCommentBody = z.object({
  content: z.string().min(1).max(3000),
}).strict();

export const ListCommentsQuery = z.object({
  postSlug: Slug,
  postType: PostType.default('posts'),
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(50),
});

export const CountQuery = z.object({
  postSlug: Slug,
  postType: PostType.default('posts'),
});

export const CountsQuery = z.object({
  slugs: z.string().transform((s, ctx) => {
    const arr = s.split(',').map(x => x.trim()).filter(Boolean);
    if (arr.length === 0) { ctx.addIssue({ code:'custom', message:'slugs required' }); return z.NEVER; }
    if (arr.length > 100) { ctx.addIssue({ code:'custom', message:'max 100' }); return z.NEVER; }
    for (const s of arr) if (!/^[a-z0-9][a-z0-9-]{0,199}$/.test(s)) {
      ctx.addIssue({ code:'custom', message:`invalid slug: ${s}` }); return z.NEVER;
    }
    return arr;
  }),
  postType: PostType.default('posts'),
});

export const MeListQuery = z.object({
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
});

export const CommentIdParam = z.object({ id: ObjectIdStr });
```

---

## HMAC slug-signing protocol spec

**Goal:** prove `(slug, postType)` came from a page the Astro build emitted.

**Env vars:**
- `SLUG_SIGNING_SECRET` — 32+ random bytes (base64url). Same value in Astro
  build env and API env. Rotate to invalidate all sigs.

**Signature format:**
- HMAC-SHA256 over UTF-8 bytes of `${postType}:${slug}`.
- base64url-encoded, truncated to 22 chars (132 bits — collision-resistant
  for spam-gate purposes; not cryptographic identity).

**Flow:**
1. Astro build (per page): compute `sig = generateSlugSig(slug, postType)`.
   Render `<CommentSection postSlug={slug} postType={postType} slugSig={sig} />`.
2. Client POST `/comments` with `{postSlug, postType, content, slugSig, ...}`.
3. API: Zod parse → `verifySlugSig(postSlug, postType, slugSig)` →
   `timingSafeEqual`. Reject 403 on mismatch.
4. GET routes (`/comments?postSlug=...`) **do not** require sig — reading
   is already public.

**Why HMAC and not JWT?** No expiry needed (slugs are permanent); no claims
needed; smaller payload (22 chars vs ~200). If you later want per-page
rate-limiting tokens, switch to a short JWT with `aud=slug:postType` and
`exp`.

---

## Migration phases

**Stage 1 — Validation only (week 1-2)**
- Add `zod` + `@hono/zod-validator` deps.
- Land schema files (`src/schemas/*.ts`) and `zValidator` middleware on every
  POST/PUT route. Keep behavior identical (no new rejections beyond what
  current code already throws).
- Add `slugSig: z.string().optional()` accepting either path.
- Add logging: when `slugSig` is missing or invalid, log
  `{event:'slug_sig_missing', slug, postType, ip}`. Do **not** reject.
- Frontend Astro pages start emitting `slugSig`.
- Remove `sanitizeContent` from write path; replace with `normalizeContent`.
  Add `sanitizeAnonName` for control/bidi strip.
- Add scheme-allowlist + ObjectId validation on `/w/:shortId` and
  `/records/:id/og` before building redirect URL.

**Stage 2 — Enforce slug signature (week 3+)**
- Watch metrics from stage 1; verify `slug_sig_missing` rate ≈ 0 from
  production traffic (excluding bots).
- Flip `REQUIRE_SLUG_SIG=true`. Schema becomes `slugSig: SlugSig` (required).
- Existing DB comments are untouched. Old un-signed clients (if any) get 403.
- Document the env var rotation procedure.

**Stage 3 — Cleanup**
- Delete dead `sanitizeContent`. Update tests that asserted the literal
  `&lt;` escaped output to assert raw storage + React-escaped DOM.
- Consider migrating existing comment `content` to un-escape the 5 chars
  (`&amp; &lt; &gt; &quot; &#39;` → originals) via a one-shot script — this
  is safe because the renderer escapes again. Without migration, old
  comments will visibly show `&amp;` text.

---

## Quick wins (can ship immediately, before full Zod migration)

1. Tighten `postSlug` regex check at `comment.service.ts:230` — reject
   non-allowlist chars.
2. Validate `parentId` regex before `parseObjectId` (Zod-grade error msg).
3. Add `Intl.Segmenter`-aware grapheme cap to `anonName`.
4. Strip control + bidi chars from `anonName` and `content`.
5. Validate `c.req.param('id')` in `/w/:shortId` and `/records/:id/og`
   against ObjectId or short-ID regex before any string interp.
6. Remove `sanitizeContent`'s HTML-escape (replace with `normalizeContent`).
   Coordinated with one-shot DB migration OR accept that old comments show
   `&amp;` until edited.
