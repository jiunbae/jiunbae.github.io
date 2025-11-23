# Technical Debt - 기술 부채 관리

> 장기적으로 해결해야 할 기술적 과제 및 아키텍처 개선사항

---

## 우선순위 분류

- 🔴 **Critical**: 즉시 해결 필요 (보안, 심각한 버그)
- 🟠 **High**: 단기 내 해결 권장 (성능, 유지보수성)
- 🟡 **Medium**: 중기 계획 필요 (확장성, 코드 품질)
- 🟢 **Low**: 장기 개선 고려 (리팩토링, 최적화)

---

## 1. 보안 및 인증

### 🔴 Critical: GitHub Token 관리 개선

**현재 문제:**
```tsx
// Admin 패널에서 평문으로 localStorage 저장
localStorage.setItem('github_token', token)
```

**위험도:**
- XSS 공격 시 토큰 노출 가능
- 토큰 만료 검증 부재
- 권한 범위 제한 없음

**해결 방안:**

**옵션 1: 서버리스 함수 프록시** (권장)
```ts
// Netlify/Vercel Functions
// api/github-proxy.ts
export default async function handler(req, res) {
  const { GITHUB_TOKEN } = process.env

  const response = await fetch('https://api.github.com/...', {
    headers: {
      Authorization: `token ${GITHUB_TOKEN}`
    }
  })

  return res.json(await response.json())
}

// Client
fetch('/api/github-proxy', {
  method: 'POST',
  body: JSON.stringify({ action: 'createFile', ... })
})
```

**옵션 2: 암호화 저장**
```tsx
import { encrypt, decrypt } from '@/utils/crypto'

// 저장 시 암호화
const encrypted = encrypt(token, secretKey)
localStorage.setItem('github_token', encrypted)

// 사용 시 복호화
const token = decrypt(localStorage.getItem('github_token'), secretKey)
```

**예상 작업량**: 3-4일
**우선순위**: 🔴 Critical

---

### 🟠 High: CSP (Content Security Policy) 추가

**현재 문제:**
- CSP 헤더 없음
- XSS 공격 방어 부족
- 외부 스크립트 제한 없음

**해결 방안:**
```tsx
// gatsby-ssr.tsx
export const onRenderBody = ({ setHtmlAttributes }) => {
  setHtmlAttributes({
    "http-equiv": "Content-Security-Policy",
    content: `
      default-src 'self';
      script-src 'self' 'unsafe-inline' https://www.googletagmanager.com;
      style-src 'self' 'unsafe-inline';
      img-src 'self' data: https:;
      font-src 'self' data:;
      connect-src 'self' https://api.github.com https://www.google-analytics.com;
      frame-ancestors 'none';
      base-uri 'self';
      form-action 'self';
    `.replace(/\s+/g, ' ').trim()
  })
}
```

**예상 작업량**: 1일
**우선순위**: 🟠 High

---

## 2. 성능 최적화

### 🟠 High: 이미지 로딩 전략 개선

**현재 문제:**
- 모든 이미지가 eager loading
- LCP (Largest Contentful Paint) 영향
- 대역폭 낭비

**해결 방안:**
```tsx
// 히어로 이미지는 eager
<GatsbyImage
  image={heroImage}
  alt={alt}
  loading="eager"
  fetchpriority="high"
/>

// 하단 이미지는 lazy
<GatsbyImage
  image={thumbnail}
  alt={alt}
  loading="lazy"
/>

// Intersection Observer로 viewport 진입 시 로드
import { useInView } from 'react-intersection-observer'

const ImageWithLazyLoad = ({ image, alt }) => {
  const { ref, inView } = useInView({
    triggerOnce: true,
    rootMargin: '200px',
  })

  return (
    <div ref={ref}>
      {inView && <GatsbyImage image={image} alt={alt} />}
    </div>
  )
}
```

**예상 작업량**: 2일
**우선순위**: 🟠 High

---

### 🟡 Medium: 번들 사이즈 최적화

**현재 문제:**
- 번들 분석 도구 없음
- 사용하지 않는 코드 제거 미흡
- Tree shaking 최적화 필요

**해결 방안:**
```bash
# 번들 분석
pnpm add -D gatsby-plugin-webpack-bundle-analyser-v2

# gatsby-config.ts
plugins: [
  {
    resolve: 'gatsby-plugin-webpack-bundle-analyser-v2',
    options: {
      devMode: false,
      analyzerMode: 'static',
      reportFilename: 'bundle-report.html',
    }
  }
]

# 빌드 후 분석
pnpm build
# public/bundle-report.html 확인
```

**최적화 대상:**
- `@uiw/react-md-editor` (Admin 페이지만 사용) → 동적 import
- `axios` → 네이티브 fetch로 대체 검토
- `date-fns` → 필요한 함수만 import

**예상 작업량**: 2-3일
**우선순위**: 🟡 Medium

---

### 🟡 Medium: 코드 스플리팅 개선

**현재 문제:**
- Admin 패널 코드가 메인 번들에 포함
- 큰 라이브러리가 초기 로딩에 포함

**해결 방안:**
```tsx
// pages/admin.tsx
import { lazy, Suspense } from 'react'

const AdminView = lazy(() => import('@/views/Admin'))

const AdminPage = () => (
  <Suspense fallback={<LoadingSpinner />}>
    <AdminView />
  </Suspense>
)

// 큰 라이브러리 동적 import
const loadEditor = async () => {
  const { default: MDEditor } = await import('@uiw/react-md-editor')
  return MDEditor
}
```

**예상 작업량**: 1-2일
**우선순위**: 🟡 Medium

---

## 3. 코드 품질 및 유지보수성

### 🟠 High: 타입 안전성 개선

**현재 문제:**
```tsx
// gatsby-config.ts:148
serialize: ({ query: { site, allMarkdownRemark } }: any) => {
  return allMarkdownRemark.nodes.map((node: any) => {
    // any 타입 남발
```

**해결 방안:**
```tsx
interface RSSNode {
  frontmatter: {
    date: string
    description?: string
    slug: string
    title: string
  }
  excerpt?: string
  html: string
}

interface RSSQuery {
  site: {
    siteMetadata: {
      siteUrl: string
    }
  }
  allMarkdownRemark: {
    nodes: RSSNode[]
  }
}

serialize: ({ query }: { query: RSSQuery }) => {
  const { site, allMarkdownRemark } = query
  return allMarkdownRemark.nodes.map((node) => {
    const description = node.frontmatter.description ?? node.excerpt ?? ''
    // ...
  })
}
```

**예상 작업량**: 1일
**우선순위**: 🟠 High

---

### 🟡 Medium: 컴포넌트 추출 및 리팩토링

**현재 문제:**
- Header 컴포넌트가 143줄로 복잡
- 여러 책임이 하나의 컴포넌트에 집중

**해결 방안:**

**AS-IS:**
```tsx
// Header.tsx (143줄)
export const Header = ({ pathname }: HeaderProps) => {
  // 스크롤 감지
  // Intersection Observer
  // 네비게이션 렌더링
  // 진행률 표시
  // ...
}
```

**TO-BE:**
```tsx
// Header.tsx (주 컴포넌트)
export const Header = ({ pathname }: HeaderProps) => {
  return (
    <header>
      <HeaderLogo />
      <Navigation pathname={pathname} />
      <HeaderActions />
      <ScrollProgress pathname={pathname} />
    </header>
  )
}

// components/HeaderLogo.tsx
export const HeaderLogo = () => { /* ... */ }

// components/Navigation.tsx
export const Navigation = ({ pathname }) => { /* ... */ }

// components/HeaderActions.tsx
export const HeaderActions = () => {
  return (
    <>
      <ProfileLink />
      <RssLink />
      <ThemeToggle />
    </>
  )
}

// components/ScrollProgress.tsx (with custom hook)
export const ScrollProgress = ({ pathname }) => {
  const { isPost, progressWidth } = useScrollProgress(pathname)
  // ...
}
```

**예상 작업량**: 2-3일
**우선순위**: 🟡 Medium

---

### 🟡 Medium: 상태 관리 개선

**현재 문제:**
- useState가 여러 컴포넌트에 분산
- Props drilling 발생 가능성
- Admin 패널 상태 관리 복잡

**해결 방안:**

**옵션 1: Context API 확장**
```tsx
// contexts/AppContext.tsx
interface AppState {
  theme: 'light' | 'dark'
  selectedTag: string | undefined
  searchQuery: string
}

export const AppProvider = ({ children }) => {
  const [state, setState] = useState<AppState>(initialState)

  const value = {
    state,
    actions: {
      setTheme,
      setSelectedTag,
      setSearchQuery,
    }
  }

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}
```

**옵션 2: Zustand 도입** (권장)
```tsx
// stores/useAppStore.ts
import create from 'zustand'

interface AppStore {
  theme: 'light' | 'dark'
  selectedTag?: string
  setTheme: (theme: 'light' | 'dark') => void
  setSelectedTag: (tag?: string) => void
}

export const useAppStore = create<AppStore>((set) => ({
  theme: 'light',
  selectedTag: undefined,
  setTheme: (theme) => set({ theme }),
  setSelectedTag: (selectedTag) => set({ selectedTag }),
}))

// 컴포넌트에서 사용
const { theme, setTheme } = useAppStore()
```

**예상 작업량**: 3-4일
**우선순위**: 🟡 Medium

---

## 4. 테스트 및 품질 보증

### 🟠 High: 테스트 인프라 구축

**현재 문제:**
- 테스트 코드 전무
- 리팩토링 시 회귀 위험
- CI/CD 품질 검증 부재

**해결 방안:**

**Phase 1: 단위 테스트 설정**
```bash
pnpm add -D vitest @testing-library/react @testing-library/jest-dom
```

```ts
// vitest.config.ts
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./test/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      exclude: [
        'node_modules/',
        'test/',
        '**/*.d.ts',
        '**/*.config.*',
      ],
    },
  },
  resolve: {
    alias: {
      '@': '/src',
    },
  },
})
```

```ts
// test/setup.ts
import '@testing-library/jest-dom'
```

**Phase 2: 우선순위 테스트 작성**
```tsx
// src/utils/__tests__/date.test.ts
import { describe, it, expect } from 'vitest'
import { formatDate } from '../date'

describe('formatDate', () => {
  it('should format date correctly', () => {
    expect(formatDate('2025-11-23')).toBe('2025년 11월 23일')
  })
})

// src/hooks/__tests__/useTag.test.ts
import { renderHook, act } from '@testing-library/react'
import { useTag } from '../useTag'

describe('useTag', () => {
  it('should select tag', () => {
    const { result } = renderHook(() => useTag(/* ... */))

    act(() => {
      result.current.clickTag('react')
    })

    expect(result.current.selectedTag).toBe('react')
  })
})

// src/components/__tests__/Tag.test.tsx
import { render, screen } from '@testing-library/react'
import { Tag } from '../Tag'

describe('Tag', () => {
  it('renders tag name', () => {
    render(<Tag name="react" />)
    expect(screen.getByText('react')).toBeInTheDocument()
  })
})
```

**Phase 3: E2E 테스트**
```bash
pnpm add -D @playwright/test
```

```ts
// e2e/home.spec.ts
import { test, expect } from '@playwright/test'

test('should filter posts by tag', async ({ page }) => {
  await page.goto('/')

  await page.click('text=react')

  const posts = page.locator('[data-testid="post-card"]')
  await expect(posts.first()).toBeVisible()
})
```

**Phase 4: CI 통합**
```yaml
# .github/workflows/test.yml
name: Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v2
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'pnpm'
      - run: pnpm install
      - run: pnpm test
      - run: pnpm test:e2e
```

**예상 작업량**: 1주
**우선순위**: 🟠 High

---

## 5. 확장성 및 아키텍처

### 🟢 Low: 다국어 지원 (i18n)

**현재 문제:**
- 한국어/영어 하드코딩
- 다국어 확장 어려움

**해결 방안:**
```bash
pnpm add gatsby-plugin-react-i18next i18next react-i18next
```

```ts
// gatsby-config.ts
{
  resolve: 'gatsby-plugin-react-i18next',
  options: {
    localeJsonSourceName: 'locale',
    languages: ['ko', 'en'],
    defaultLanguage: 'ko',
    siteUrl: 'https://blog.jiun.dev',
    i18nextOptions: {
      interpolation: {
        escapeValue: false,
      },
      keySeparator: false,
      nsSeparator: false,
    },
  },
}
```

```json
// locales/ko/common.json
{
  "nav.posts": "포스트",
  "nav.notes": "노트",
  "nav.reviews": "리뷰",
  "post.readMore": "더 읽기",
  "post.minuteRead": "{{minutes}}분 읽기"
}
```

```tsx
// 컴포넌트에서 사용
import { useTranslation } from 'react-i18next'

const Navigation = () => {
  const { t } = useTranslation()

  return (
    <nav>
      <Link to="/">{t('nav.posts')}</Link>
      <Link to="/notes/">{t('nav.notes')}</Link>
    </nav>
  )
}
```

**예상 작업량**: 1주
**우선순위**: 🟢 Low

---

### 🟢 Low: 디자인 시스템 구축

**현재 문제:**
- 컴포넌트 스타일 일관성 부족
- 디자인 토큰 체계화 부족
- Storybook 없음

**해결 방안:**

**Phase 1: 디자인 토큰 정의**
```scss
// styles/tokens/_colors.scss
:root {
  // Brand colors
  --color-brand-primary: #1E88E5;
  --color-brand-secondary: #FF6F00;

  // Semantic colors
  --color-text-primary: var(--gray-1);
  --color-text-secondary: var(--gray-4);
  --color-background-primary: var(--background);
  --color-background-secondary: var(--article-background);

  // State colors
  --color-success: #4CAF50;
  --color-warning: #FF9800;
  --color-error: #F44336;
  --color-info: #2196F3;
}

// styles/tokens/_spacing.scss
:root {
  --space-1: 0.25rem;  // 4px
  --space-2: 0.5rem;   // 8px
  --space-3: 0.75rem;  // 12px
  --space-4: 1rem;     // 16px
  --space-5: 1.5rem;   // 24px
  --space-6: 2rem;     // 32px
  --space-8: 3rem;     // 48px
  --space-10: 4rem;    // 64px
}

// styles/tokens/_typography.scss
:root {
  --font-size-xs: 0.75rem;   // 12px
  --font-size-sm: 0.875rem;  // 14px
  --font-size-md: 1rem;      // 16px
  --font-size-lg: 1.125rem;  // 18px
  --font-size-xl: 1.25rem;   // 20px
  --font-size-2xl: 1.5rem;   // 24px
  --font-size-3xl: 2rem;     // 32px
  --font-size-4xl: 2.5rem;   // 40px
}
```

**Phase 2: 공통 컴포넌트 표준화**
```tsx
// components/design-system/Button/Button.tsx
interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  children: React.ReactNode
  onClick?: () => void
  disabled?: boolean
  icon?: React.ReactNode
}

export const Button = ({
  variant = 'primary',
  size = 'md',
  children,
  ...props
}: ButtonProps) => {
  return (
    <button
      className={clsx(
        styles.button,
        styles[variant],
        styles[size],
      )}
      {...props}
    >
      {children}
    </button>
  )
}

// Button.module.scss
.button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  border-radius: 4px;
  font-weight: 600;
  transition: all 0.15s ease;

  &:focus-visible {
    outline: 2px solid var(--color-brand-primary);
    outline-offset: 2px;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
}

.primary {
  background-color: var(--color-brand-primary);
  color: white;

  &:hover {
    background-color: var(--color-brand-primary-dark);
  }
}

.secondary {
  background-color: transparent;
  border: 1px solid var(--color-brand-primary);
  color: var(--color-brand-primary);

  &:hover {
    background-color: var(--color-brand-primary);
    color: white;
  }
}

.ghost {
  background-color: transparent;
  color: var(--color-text-primary);

  &:hover {
    background-color: var(--gray-6);
  }
}

.sm {
  padding: var(--space-2) var(--space-3);
  font-size: var(--font-size-sm);
}

.md {
  padding: var(--space-3) var(--space-5);
  font-size: var(--font-size-md);
}

.lg {
  padding: var(--space-4) var(--space-6);
  font-size: var(--font-size-lg);
}
```

**Phase 3: Storybook 설정**
```bash
pnpm dlx storybook@latest init
```

```tsx
// Button.stories.tsx
import type { Meta, StoryObj } from '@storybook/react'
import { Button } from './Button'

const meta: Meta<typeof Button> = {
  title: 'Design System/Button',
  component: Button,
  argTypes: {
    variant: {
      control: 'select',
      options: ['primary', 'secondary', 'ghost'],
    },
    size: {
      control: 'select',
      options: ['sm', 'md', 'lg'],
    },
  },
}

export default meta
type Story = StoryObj<typeof Button>

export const Primary: Story = {
  args: {
    children: 'Primary Button',
    variant: 'primary',
  },
}

export const Secondary: Story = {
  args: {
    children: 'Secondary Button',
    variant: 'secondary',
  },
}

export const WithIcon: Story = {
  args: {
    children: 'With Icon',
    icon: <Icon />,
  },
}
```

**예상 작업량**: 2주
**우선순위**: 🟢 Low

---

## 6. 모니터링 및 관찰성

### 🟡 Medium: 에러 모니터링

**현재 문제:**
- 프로덕션 에러 추적 불가
- 사용자 오류 보고 메커니즘 없음

**해결 방안:**
```bash
pnpm add @sentry/gatsby
```

```ts
// gatsby-config.ts
{
  resolve: '@sentry/gatsby',
  options: {
    dsn: process.env.SENTRY_DSN,
    sampleRate: 1.0,
    tracesSampleRate: 0.1,
    environment: process.env.NODE_ENV,
  },
}

// gatsby-browser.tsx
import * as Sentry from '@sentry/gatsby'

export const onClientEntry = () => {
  Sentry.init({
    dsn: process.env.GATSBY_SENTRY_DSN,
    integrations: [
      new Sentry.BrowserTracing(),
      new Sentry.Replay(),
    ],
    tracesSampleRate: 0.1,
    replaysSessionSampleRate: 0.1,
    replaysOnErrorSampleRate: 1.0,
  })
}
```

**예상 작업량**: 1일
**우선순위**: 🟡 Medium

---

### 🟡 Medium: Web Vitals 모니터링

**현재 문제:**
- Core Web Vitals 측정 없음
- 성능 저하 감지 불가

**해결 방안:**
```tsx
// src/components/WebVitals.tsx
import { useEffect } from 'react'

export const WebVitals = () => {
  useEffect(() => {
    if ('web-vital' in window && typeof window.webVitals !== 'undefined') {
      const { getCLS, getFID, getFCP, getLCP, getTTFB } = window.webVitals

      getCLS(sendToAnalytics)
      getFID(sendToAnalytics)
      getFCP(sendToAnalytics)
      getLCP(sendToAnalytics)
      getTTFB(sendToAnalytics)
    }
  }, [])

  return null
}

function sendToAnalytics(metric: any) {
  if (typeof window.gtag !== 'undefined') {
    window.gtag('event', metric.name, {
      value: Math.round(metric.name === 'CLS' ? metric.value * 1000 : metric.value),
      event_category: 'Web Vitals',
      event_label: metric.id,
      non_interaction: true,
    })
  }
}

// gatsby-browser.tsx
import { WebVitals } from './src/components/WebVitals'

export const wrapPageElement = ({ element }) => (
  <>
    <WebVitals />
    {element}
  </>
)
```

**예상 작업량**: 1일
**우선순위**: 🟡 Medium

---

## 7. 개발 경험 (DX) 개선

### 🟢 Low: Pre-commit Hooks

**해결 방안:**
```bash
pnpm add -D husky lint-staged
pnpm exec husky init
```

```json
// package.json
{
  "lint-staged": {
    "*.{ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{scss,css}": [
      "prettier --write"
    ]
  }
}
```

```bash
# .husky/pre-commit
pnpm lint-staged
pnpm typecheck
```

**예상 작업량**: 1시간
**우선순위**: 🟢 Low

---

### 🟢 Low: Prettier 설정

**해결 방안:**
```bash
pnpm add -D prettier
```

```json
// .prettierrc
{
  "semi": false,
  "singleQuote": true,
  "trailingComma": "es5",
  "tabWidth": 2,
  "printWidth": 100,
  "arrowParens": "always"
}
```

**예상 작업량**: 30분
**우선순위**: 🟢 Low

---

## 타임라인 제안

### Phase 1: Foundation (1-2주)
- ✅ CSP 추가
- ✅ 타입 안전성 개선
- ✅ 테스트 인프라 구축

### Phase 2: Quality (2-3주)
- ✅ 이미지 로딩 최적화
- ✅ 번들 사이즈 최적화
- ✅ 에러 모니터링 추가
- ✅ Web Vitals 모니터링

### Phase 3: Refactoring (3-4주)
- ✅ 컴포넌트 리팩토링
- ✅ 상태 관리 개선
- ✅ 코드 스플리팅 개선

### Phase 4: Enhancement (장기)
- ✅ 디자인 시스템 구축
- ✅ 다국어 지원
- ✅ DX 개선

---

## 추적 및 모니터링

### 기술 부채 메트릭

**측정 지표:**
- TypeScript strict mode violations: `tsc --noEmit` 에러 개수
- ESLint warnings: `eslint . --max-warnings 0`
- Test coverage: `vitest --coverage` 목표 80%+
- Bundle size: `gatsby build` 후 분석, 목표 200KB 이하 (gzipped)
- Lighthouse score: 목표 95+ (Performance, Accessibility, Best Practices, SEO)

**정기 리뷰:**
- 월 1회 기술 부채 리뷰 회의
- 분기 1회 아키텍처 리뷰
- 매 릴리스 전 품질 체크

---

**마지막 업데이트**: 2025-11-23
**다음 리뷰**: 2025-12-23
