# jiun.dev

Gatsby 기반 개인 블로그

## Tech Stack

- **Framework**: Gatsby 5
- **Language**: TypeScript
- **Styling**: SCSS Modules
- **Package Manager**: pnpm

## Getting Started

### Prerequisites

- Node.js >= 18
- pnpm >= 9

### Installation

```bash
pnpm install
```

### Development

```bash
pnpm develop
```

http://localhost:8000 에서 확인

### Build

```bash
pnpm build
```

### Deploy

```bash
pnpm deploy
```

GitHub Pages로 배포

## Project Structure

```
├── src/
│   ├── components/     # 재사용 컴포넌트
│   ├── contexts/       # React Context
│   ├── layouts/        # 레이아웃
│   ├── pages/          # 페이지
│   ├── styles/         # SCSS 스타일
│   ├── templates/      # 페이지 템플릿
│   ├── utils/          # 유틸리티 함수
│   └── views/          # 뷰 컴포넌트
├── static/             # 정적 파일
├── scripts/            # 빌드 스크립트
├── gatsby-config.ts    # Gatsby 설정
├── gatsby-node.ts      # Gatsby Node API
└── gatsby-ssr.tsx      # SSR 설정
```

## Features

### OG Image Generation

빌드 타임에 각 포스트의 OG Image를 자동 생성합니다.

- SVG → PNG 변환 (resvg)
- 제목, 요약, 날짜 포함
- 1200x630px (표준 OG 크기)

### Syntax Highlighting

PrismJS 기반 코드 하이라이팅

### Mermaid Diagrams

마크다운에서 Mermaid 다이어그램 지원

### RSS Feed

`/rss.xml`에서 RSS 피드 제공

### Sitemap

`/sitemap.xml`에서 사이트맵 제공

## Scripts

| Script | Description |
|--------|-------------|
| `pnpm develop` | 개발 서버 실행 |
| `pnpm build` | 프로덕션 빌드 |
| `pnpm serve` | 빌드 결과 로컬 서빙 |
| `pnpm clean` | 캐시 및 빌드 결과 삭제 |
| `pnpm deploy` | GitHub Pages 배포 |
| `pnpm typecheck` | TypeScript 타입 체크 |
| `pnpm lint` | ESLint 실행 |

## License

MIT
