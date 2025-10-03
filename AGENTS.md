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
