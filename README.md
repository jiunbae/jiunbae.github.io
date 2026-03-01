<div align="center">

# jiun.dev

**Personal blog & creative playground**

[![Astro](https://img.shields.io/badge/Astro-5.x-BC52EE?logo=astro&logoColor=white)](https://astro.build)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Live Site](https://jiun.dev) · [Blog Posts](https://jiun.dev/) · [Playground](https://jiun.dev/playground/) · [Tools](https://jiun.dev/tools/)

</div>

---

## Overview

A fast, statically-generated personal site built with **Astro 5** and **React 18** islands. Features a tech blog, interactive creative coding demos, and browser-based utility tools — all shipped as zero-JS static pages with selective client hydration.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | [Astro 5](https://astro.build) (Static Site Generation) |
| Islands | [React 18](https://react.dev) (`client:only` / `client:idle`) |
| Language | TypeScript 5 |
| Styling | SCSS Modules + CSS Custom Properties |
| Content | Astro Content Collections (MDX) |
| Code Blocks | [Expressive Code](https://expressive-code.com) |
| Search | [Fuse.js](https://www.fusejs.io) (client-side fuzzy search) |
| OG Images | Raw SVG + [@resvg/resvg-js](https://github.com/nicolo-ribaudo/resvg-js) |
| 3D / WebGL | [Three.js](https://threejs.org) |
| Media | [@ffmpeg/ffmpeg](https://ffmpegwasm.netlify.app) (WASM) |
| Package Manager | pnpm 9 |

## Features

### Blog & Notes
- MDX content with syntax highlighting, Mermaid diagrams, and LaTeX math
- Tag-based filtering with animated transitions
- Full-text search via `Cmd/Ctrl+K`
- Auto-generated OG images (1200x630, Korean font support)
- RSS feed at `/rss.xml`

### Interactive Playground
14 creative coding demos rendered as React islands:

| Demo | Tech |
|------|------|
| Cyber Flowers | Three.js · GLSL shaders |
| Particle Galaxy | Three.js · BufferGeometry |
| Audio Visualizer | Three.js · Web Audio API |
| Fluid Simulation | Canvas 2D · Navier-Stokes |
| Generative Art | Canvas 2D · Perlin noise |
| Raymarching | Canvas 2D · SDF rendering |
| Shader Art | WebGL · Fragment shaders |
| Terrain Generator | Three.js · Heightmap |
| Mandelbrot Set | Canvas 2D · Complex math |
| Boids | Canvas 2D · Flocking algorithm |
| Cellular Automata | Canvas 2D · Conway's GoL |
| Physics Sandbox | Canvas 2D · Verlet integration |
| Sorting Lab | Canvas 2D · Algorithm visualization |
| Wave Interference | Canvas 2D · Wave physics |

### Browser Tools
Client-side file converters powered by FFmpeg WASM — no server uploads:

- **Image Converter** — PNG, JPEG, WebP, GIF, BMP, TIFF, PDF
- **Audio Converter** — MP3, WAV, AAC, OGG, FLAC, M4A
- **Video Converter** — MP4, WebM, AVI, MOV, MKV, GIF

## Project Structure

```
src/
├── content/          # MDX posts, notes, reviews (Content Collections)
├── components/       # Shared components (SearchModal, etc.)
├── data/             # Static data (playground items, tool definitions)
├── layouts/          # Layout.astro, BaseHead.astro
├── pages/
│   ├── index.astro           # Home (post listing)
│   ├── about.astro           # About page
│   ├── playground.astro      # Playground index
│   ├── posts/[...slug].astro # Blog post detail
│   ├── notes/[...slug].astro # Note detail
│   ├── tools/                # Converter tools
│   ├── og/[...slug].png.ts   # OG image generation endpoint
│   └── rss.xml.ts            # RSS feed
├── plugins/          # Rehype plugins (mermaid, etc.)
├── styles/           # Global SCSS (fonts, variables, reset)
├── utils/            # Utilities (ffmpeg wrapper, etc.)
└── views/            # React island components
    ├── CyberFlowers/
    ├── ParticleGalaxy/
    ├── ImageConverter/
    └── ...
```

## Getting Started

### Prerequisites

- Node.js >= 18
- pnpm >= 9

### Install & Run

```bash
# Install dependencies
pnpm install

# Start dev server
pnpm dev

# Build for production
pnpm build

# Preview production build
pnpm preview
```

Dev server runs at `http://localhost:4321`.

## Scripts

| Script | Description |
|--------|-------------|
| `pnpm dev` | Start development server |
| `pnpm build` | Production build (static output) |
| `pnpm preview` | Serve production build locally |

## Architecture Decisions

- **Astro islands** — Heavy interactive components (Three.js, Canvas, FFmpeg) use `client:only="react"` to avoid SSR overhead. Static pages ship zero JavaScript.
- **Raw SVG for OG images** — Uses handcrafted SVG templates + Resvg instead of Satori for pixel-perfect match with the original design (66px title, 40px body, dark slate gradient).
- **Content Collections** — Type-safe frontmatter validation with Zod schemas. Posts, notes, and reviews each have their own collection.
- **SCSS Modules** — Component-scoped styles with CSS custom properties for theming (light/dark mode).

## License

MIT
