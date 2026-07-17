# Repository Guidelines

## Project Structure & Module Organization
This site runs on Astro 6 with React islands and TypeScript. `src/pages` defines file-based routes (including `posts/`, `notes/`, `reviews/`, `tools/`, `status/`, `design/`, `og/`, `thumbs/` subfolders and standalone playground routes like `boids.astro`, `fluid-sim.astro`). Reusable UI, hooks, and interactive views live in `src/components`, `src/layouts`, and `src/views` (one folder per playground demo, e.g. `AudioConverter`, `Mandelbrot`, `Raymarching`). Styles sit under `src/styles` (`abstracts/`, `base/`, `pages/`, `utils/`) as SCSS partials. Structured data (services, playground entries, tools, about info) lives in `src/data/*.ts`/`.json`.

Markdown/MDX content lives in `src/content/`, split by collection: `posts/`, `notes/`, `reviews/`, `incidents/`. Each entry is a folder — `src/content/posts/YYYY-MM-DD-slug/index.md` — and collection schemas (frontmatter validation) are defined centrally in `src/content.config.ts`. The `permalink` frontmatter field determines the final URL slug, independent of the folder name. Static assets go to `public/`; Astro emits the production build into `dist/` (gitignored, do not edit manually).

## Build, Test, and Development Commands
Use `pnpm install` once, then run workflows with pnpm scripts:
- `pnpm run dev`: local dev server with hot reload (Astro dev).
- `pnpm run build`: production build; run before PRs touching runtime code, styles, or content.
- `pnpm run preview`: preview the last build at `http://localhost:4321` (audit/visual scripts serve it on `4322`).
- `pnpm run lint:style`: stylelint over `src/**/*.scss` (no ESLint in this repo — TypeScript is checked via `astro check`/editor tooling against `tsconfig.json`).
- `pnpm run audit:tokens`: scans `src/{pages,components,layouts,views,styles}` for raw spacing/radius/motion/color/shadow values vs. design-token usage; appends to `scripts/audit-history.jsonl`.
- `pnpm run audit:a11y`: runs axe-core (via Playwright) against the built site's key routes; writes `scripts/axe-latest.json` + per-page reports under `scripts/axe-reports/`.
- `pnpm run visual:baseline` / `pnpm run visual:check`: Playwright screenshots across pages × {light, dark} × {mobile, desktop}; `baseline` captures reference PNGs into `scripts/visual-baseline/`, `check` diffs the current build against them.
- `pnpm run score` / `pnpm run score:show`: records/shows cumulative persona-review scores (see `.agents/` below).

### When to run the validation chain
Any change touching design tokens, layout, or visual styling should go through: `pnpm run build` → `pnpm run audit:tokens` → `pnpm run audit:a11y` → `pnpm run visual:check`. Only after the change is reviewed/approved, refresh the reference snapshots with `pnpm run visual:baseline` (baseline is the accepted-state snapshot, so don't regenerate it before approval or you'll mask regressions).

## Coding Style & Naming Conventions
No ESLint config is present in this repo; TypeScript strictness comes from `tsconfig.json` (`astro/tsconfigs/strict`) and SCSS is linted with stylelint (`.stylelintrc.json`, `stylelint-config-standard-scss`). Follow existing conventions: single quotes, no semicolons, two-space indent. Components/layouts use `PascalCase`, utilities `camelCase`, and content folders `kebab-case` (`YYYY-MM-DD-slug/`). Keep React components functional. Re-run `pnpm run lint:style` after SCSS changes.

## Testing Guidelines
There is no unit test suite; rely on `pnpm run build` for regression coverage, `pnpm run preview` to inspect generated pages, and the audit/visual scripts above for design-system and accessibility regressions. Mirror existing file placement when adding content (`src/content/<collection>/YYYY-MM-DD-slug/index.md`), and keep frontmatter complete per `src/content.config.ts` (`title`, `date`, `permalink`, `description`, `tags`, `published`, plus `rating` for reviews or the incident-specific fields for `incidents/`).

## Commit & Pull Request Guidelines
Commits in history are short imperative statements, increasingly with a `type(scope):`/`type:` prefix (e.g., `feat(nav): expose /design via footer`, `fix(a11y): touch targets, reduced-motion transform guard`, `docs(.agents): codify 4-reviewer personas`) alongside plain ones (`Add Calex App Store pages`). Follow whichever pattern fits the change, group related work, and reference issues with `(#id)` when relevant. PRs should explain the change scope, call out impacted routes or content directories, and attach before/after screenshots for visual tweaks. Verify `pnpm run build` locally (plus the audit/visual chain for design changes) and note any manual validation steps in the PR description.

## Content & Deployment Notes
Deployment is automated: `.github/workflows/astro.yml` builds with `astro build --site https://jiun.dev` and publishes `dist/` to GitHub Pages via `actions/deploy-pages` on every push to `main` (PRs get a build-only check). There is no manual `gh-pages`/`pnpm run deploy` step anymore — merging to `main` ships the site. Use feature branches; avoid committing build artifacts (`dist/`, `.astro/`). Secrets live in your local shell—do not hardcode keys in config files.

## Design Review Personas
`.agents/` archives the cold-review process used for design-system changes: `.agents/personas/{A,B,C,D}-*.md` define four reviewer lenses (DS adoption/Atomic Design, Nielsen heuristics + Korean readability, WCAG/ARIA accessibility, Gestalt/IA/grid), and `.agents/reviews/R<N>.md` record per-round scores and findings (append-only). When running a review round, spawn the four personas in parallel, log scores with `pnpm run score add R<N> <A> <B> <C> <D> "<note>"`, and cite the `audit:tokens`/`audit:a11y` output in the round notes. See `.agents/README.md` for the full protocol.

## Writing Style Guidelines (블로그 글쓰기 가이드라인)

### 피해야 할 패턴 (LLM 생성 글의 특징)
1. **템플릿 헤더 금지**: `> **목표**: ... > **결과**: ... > **독자**: ...` 형식 사용하지 않기
2. **과도한 구조화 금지**: 모든 문단을 `## 1. 제목`, `### 1.1 소제목` 식으로 번호 매기지 않기
3. **불필요한 코드 블록 금지**: 설명용 pseudo-code나 예시 코드를 과도하게 넣지 않기
4. **마케팅 용어 자제**: "폭발적인 성장", "핵심 역량", "혁신적인" 같은 수식어 최소화
5. **SEO 키워드 스터핑 금지**: frontmatter에 keywords, metaDescription 남발하지 않기
6. **과도한 리스트 금지**: 모든 내용을 bullet point로 나열하지 않기

### 지향해야 할 스타일
1. **개인적 톤**: 1인칭 시점으로 경험 공유, 구어체 허용 ("~했다", "~인 것 같다")
2. **간결함**: 150-300줄 내외, 핵심만 전달
3. **유머/위트**: 적절한 농담이나 자기비하 허용 ("삽질했다", "뻘짓이었다")
4. **구체적 경험**: 추상적 설명 대신 실제 겪은 사례 중심
5. **자연스러운 흐름**: 목차 없이도 읽히는 문단 구성
6. **솔직함**: 실패 경험, 모르는 것 인정, 불확실성 표현

### 좋은 글의 예시 구조
```markdown
---
title: "제목"
date: 2024-01-01
permalink: /post-slug
tags: [ai, dev]
published: true
---

(도입 - 왜 이 글을 쓰게 됐는지 1-2문장)

## 배경/상황 설명
(어떤 문제가 있었는지, 무엇을 하려고 했는지)

## 본론
(시도한 것, 결과, 배운 점 - 자연스러운 서술체로)

## 마무리
(짧은 정리, 후기, 느낀점)
```

### 코드 블록 사용 원칙
- 실제로 동작하는 코드만 포함
- 설명을 위한 pseudo-code는 최소화
- 핵심 로직만 발췌, 전체 구현은 GitHub 링크로 대체
- 코드보다 "왜" 그렇게 했는지가 더 중요

### 피해야 할 문장 예시
- ❌ "본 글에서는 ~에 대해 다루겠습니다"
- ❌ "이 기술은 혁신적인 패러다임 전환을 가져왔습니다"
- ❌ "결론적으로, ~는 매우 중요합니다"
- ✅ "최근 ~를 써봤는데 괜찮았다"
- ✅ "처음엔 삽질을 좀 했다"
- ✅ "솔직히 아직 잘 모르겠다"
