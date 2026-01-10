# Repository Guidelines

## Project Structure & Module Organization
This site runs on Gatsby 5 with React and TypeScript. Reusable UI and hooks live in `src/components`, `src/views`, and `src/utils`, while `src/pages` and `src/templates` define page shells consumed by Gatsby. Styles sit under `src/styles` and component-scoped SCSS modules. Markdown content and JSON data live in `contents/` (`posts/`, `notes/`, `temp/`, `data/`). Static assets go to `static/`; Gatsby emits production bundles into `public/` (do not edit manually). Gatsby configuration is centralized in `gatsby-config.ts` and `gatsby-node.ts`.

## Build, Test, and Development Commands
Use `pnpm install` once, then run workflows with pnpm scripts:
- `pnpm run develop`: local dev server with hot reload.
- `pnpm run build`: production build; run before PRs touching runtime code or content.
- `pnpm run serve`: preview the last build at `http://localhost:9000`.
- `pnpm run clean`: reset Gatsby caches when data changes fail to surface.
- `pnpm run typecheck`: strict TypeScript validation with the project tsconfig.
- `pnpm run lint`: ESLint autofix; pair with a manual review of remaining warnings.
- `pnpm run convert`: convert labeled GitHub issues into markdown files under `contents/`.

## Coding Style & Naming Conventions
Follow the ESLint config (`eslint.config.mjs`): single quotes, no semicolons, avoid dangling commas, and ignore `_`-prefixed unused params. Components and layouts use `PascalCase`, utilities `camelCase`, and markdown filenames `kebab-case.md`. Indent with two spaces and keep React components functional. Re-run `pnpm run lint` after formatting changes.

## Testing Guidelines
Automated tests are not yet wired; rely on `pnpm run build` for regression coverage and inspect generated pages under `pnpm run serve`. Mirror existing file placement when adding content, and keep frontmatter complete (`title`, `date`, `slug`, `description`, media refs). If you introduce runtime utilities, add usage examples in the corresponding markdown post.

## Commit & Pull Request Guidelines
Commits in history are short imperative statements (e.g., `Update components`, `Fix og image (#9)`). Follow that pattern, grouping related changes and referencing issues with `(#id)` when relevant. PRs should explain the change scope, call out impacted routes or content directories, and attach before/after screenshots for visual tweaks. Verify `pnpm run build` locally and note any manual validation steps in the PR description.

## Content & Deployment Notes
Content updates deploy via `pnpm run deploy`, which cleans, rebuilds, and pushes `public/` with `gh-pages`. Use feature branches; avoid committing build artifacts. Secrets live in your local shell—do not hardcode keys in config files.

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
slug: /post-slug
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
