# Quick Wins - 빠르게 적용 가능한 개선사항

> 개발 시간 1-2시간 이내로 즉시 적용 가능한 개선사항 모음

---

## 1. 성능 최적화 (30분)

### 전역 transition 제거

**파일**: `src/styles/base/_global.scss`

```diff
-* {
-  transition: background-color 0.15s ease, color 0.15s ease;
-}
+body,
+a,
+button,
+.header,
+[class*="Card"] {
+  transition: background-color 0.15s ease, color 0.15s ease;
+}
```

**효과**: 리페인트 성능 개선, 부드러운 스크롤

---

### 스크롤 이벤트 passive 옵션 추가

**파일**: `src/layouts/components/Header/Header.tsx`

```diff
useEffect(() => {
  const handleScroll = () => {
    setIsShrink(window.scrollY > 0)
  }
-  window.addEventListener('scroll', handleScroll)
+  window.addEventListener('scroll', handleScroll, { passive: true })
  return () => window.removeEventListener('scroll', handleScroll)
}, [])
```

**효과**: 스크롤 성능 개선

---

## 2. 접근성 개선 (45분)

### ARIA 속성 추가

**파일**: `src/layouts/components/Header/Header.tsx`

```diff
<nav className={styles.navigation}>
+  aria-label="Main navigation"
  {navLinks.map(link => (
    <Link
      key={link.label}
      to={link.to}
      state={link.state}
      className={clsx(styles.link, { [styles.activeLink]: isActivePath(link.to) })}
+     aria-current={isActivePath(link.to) ? "page" : undefined}
    >
      {link.label}
    </Link>
  ))}
</nav>
```

**파일**: `src/components/FloatingButton/FloatingButton.tsx`

```diff
<button
  className={styles.button}
  onClick={scrollToTop}
+  aria-label="Scroll to top"
+  aria-hidden={!isVisible}
>
  <ArrowIcon />
</button>
```

**효과**: 스크린 리더 지원, 키보드 접근성 개선

---

### 색상 대비 개선

**파일**: `src/styles/base/_global.scss`

```diff
:root {
  --gray-1: #121212;
  --gray-2: #282828;
  --gray-3: #3f3f3f;
  --gray-4: #575757;
-  --gray-5: #717171;
+  --gray-5: #666666;
  --gray-6: #8b8b8b;
}
```

**효과**: WCAG AA 기준 충족

---

## 3. UX 개선 (30분)

### 링크 가시성 개선

**파일**: `src/styles/base/_global.scss`

```scss
// 추가
a {
  color: var(--primary-c1);
  text-decoration-color: rgba(100, 181, 246, 0.3);
  text-underline-offset: 2px;
  transition: text-decoration-color 0.15s ease;

  &:hover {
    text-decoration-color: var(--primary-c1);
  }

  &:focus-visible {
    outline: 2px solid var(--primary-c1);
    outline-offset: 2px;
    border-radius: 2px;
  }
}
```

**효과**: 링크 인식 개선, 키보드 포커스 가시화

---

### 버튼 클릭 피드백 추가

**파일**: `src/styles/base/_global.scss`

```scss
// 추가
button,
[role="button"] {
  &:active {
    transform: scale(0.98);
  }

  &:focus-visible {
    outline: 2px solid var(--primary-c1);
    outline-offset: 2px;
  }
}
```

**효과**: 인터랙션 피드백 개선

---

## 4. 모바일 개선 (45분)

### 터치 타겟 크기 확대

**파일**: `src/layouts/components/Header/Header.module.scss`

```diff
.icon {
  width: 26px;
  height: 26px;
+  padding: 9px; // 44x44px 터치 영역 확보
  color: var(--icon-color);
  stroke: var(--icon-color);
  transition: width 0.3s, height 0.3s;
}
```

**효과**: 모바일 탭 정확도 개선

---

### 모바일 패딩 확대

**파일**: `src/views/Home/Home.module.scss`

```diff
.wrapper {
-  padding: 20px;
+  padding: 24px 20px;

  @include tabletAndUp {
    padding: 20px 60px;
  }
}
```

**효과**: 가독성 개선

---

## 5. SEO 개선 (15분)

### RSS 피드 description 개선

**파일**: `gatsby-config.ts`

```diff
feeds: [
  {
    serialize: ({ query: { site, allMarkdownRemark } }: any) => {
      return allMarkdownRemark.nodes.map((node: any) => {
-        const description = node.frontmatter.description ?? node.excerpt ?? ''
+        const description = node.frontmatter.description || node.excerpt || '새로운 포스트를 확인하세요'
        return Object.assign({}, node.frontmatter, {
          description,
          date: new Date(node.frontmatter.date),
          url: `${site.siteMetadata.siteUrl}/posts${node.frontmatter.slug}`,
          guid: `${site.siteMetadata.siteUrl}/posts${node.frontmatter.slug}`,
          custom_elements: [{ 'content:encoded': node.html }]
        })
      })
    },
```

**효과**: RSS 리더 가독성 개선

---

## 6. 브랜드 개선 (30분)

### 다크 모드 이미지 밝기 조정

**파일**: `src/styles/base/_global.scss`

```scss
// 추가
[data-theme='dark'] {
  img:not([class*="logo"]):not([class*="icon"]) {
    opacity: 0.9;
    transition: opacity 0.2s;

    &:hover {
      opacity: 1;
    }
  }
}
```

**효과**: 다크 모드 가독성 개선

---

### 선택 영역 색상 개선

**파일**: `src/styles/base/_global.scss`

```diff
::selection {
-  background-color: #15af7333;
+  background-color: var(--primary-c1);
+  color: #ffffff;
}

+[data-theme='dark'] ::selection {
+  background-color: var(--primary-c2);
+  color: #000000;
+}
```

**효과**: 브랜드 일관성 개선

---

## 7. 코드 품질 (30분)

### 사용하지 않는 의존성 제거

```bash
pnpm remove svg-react-loader
```

**효과**: 번들 사이즈 감소

---

### ESLint 설정 추가

**파일**: `.eslintrc.js` (또는 `eslint.config.js`)

```js
module.exports = {
  rules: {
    'react-hooks/exhaustive-deps': 'warn',
    'no-console': ['warn', { allow: ['warn', 'error'] }],
    '@typescript-eslint/no-explicit-any': 'warn',
  }
}
```

**효과**: 코드 품질 향상

---

## 적용 순서

1. **성능 최적화** (30분) - 즉시 효과
2. **접근성 개선** (45분) - SEO 및 법적 준수
3. **UX 개선** (30분) - 사용자 만족도
4. **모바일 개선** (45분) - 모바일 트래픽 대응
5. **SEO 개선** (15분) - 검색 노출
6. **브랜드 개선** (30분) - 심미성
7. **코드 품질** (30분) - 유지보수

**총 소요 시간**: 약 3-4시간

---

## 테스트 체크리스트

각 개선사항 적용 후 확인:

- [ ] 빌드 성공 (`pnpm build`)
- [ ] 타입 에러 없음 (`pnpm typecheck`)
- [ ] Lint 통과 (`pnpm lint`)
- [ ] 로컬 개발 서버 정상 작동 (`pnpm develop`)
- [ ] 모바일 뷰 확인 (Chrome DevTools)
- [ ] 다크 모드 확인
- [ ] Lighthouse 점수 개선 확인

---

## 다음 단계

Quick Wins 완료 후:
1. `comprehensive-analysis.md`의 High Priority 항목 검토
2. 개선사항별 이슈 생성
3. 우선순위에 따라 구현

---

**참고**: 이 문서의 모든 변경사항은 git branch를 만들어 개별적으로 적용하고 테스트하는 것을 권장합니다.

```bash
git checkout -b improve/quick-wins
# 변경사항 적용
git add .
git commit -m "chore: apply quick wins improvements"
git push origin improve/quick-wins
# PR 생성 및 리뷰
```
