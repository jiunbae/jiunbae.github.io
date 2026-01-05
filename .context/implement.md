# 페이지별 JSON-LD 적용 명세

## 목표
각 페이지에 적절한 JSON-LD 구조화 데이터 추가

## 전제 조건
- `src/components/JsonLd.tsx` 컴포넌트가 존재해야 함
- 확장된 `Seo.tsx` 컴포넌트가 있어야 함

## 작업 항목

### 1. GraphQL 쿼리 수정 - ISO 날짜 필드 추가

#### Post.tsx (src/templates/Post.tsx)
```graphql
frontmatter {
  date(formatString: "YY.MM.DD")
  dateISO: date(formatString: "YYYY-MM-DD")  # 추가
  # ... 기존 필드
}
```

#### Note.tsx (src/templates/Note.tsx)
```graphql
frontmatter {
  date(formatString: "YY.MM.DD")
  dateISO: date(formatString: "YYYY-MM-DD")  # 추가
  # ... 기존 필드
}
```

#### Review.tsx (src/templates/Review.tsx)
```graphql
frontmatter {
  date(formatString: "YYYY년 MM월 DD일")
  dateISO: date(formatString: "YYYY-MM-DD")  # 추가
  # ... 기존 필드
}
```

### 2. 페이지별 JSON-LD 추가

#### Home (src/views/Home/Home.tsx)
- WebSite 스키마 추가
- SearchAction 포함

```tsx
import { JsonLd, createWebSiteSchema } from '@/components'

export const Head = (...) => {
  const webSiteSchema = createWebSiteSchema({
    siteUrl: 'https://blog.jiun.dev',
    siteName: "Jiunbae's Blog",
    description: site?.siteMetadata.description || ''
  })

  return (
    <>
      <Seo ... type="website" />
      <JsonLd data={webSiteSchema} />
    </>
  )
}
```

#### Post (src/views/Post/Post.tsx)
- BlogPosting 스키마
- BreadcrumbList 스키마

```tsx
import { JsonLd, createBlogPostingSchema, createBreadcrumbSchema } from '@/components'

export const Head = (...) => {
  const blogPostingSchema = createBlogPostingSchema({
    title,
    description,
    datePublished: frontmatter.dateISO,
    url: `https://blog.jiun.dev${pageUrl}`,
    image: heroImage ? `https://blog.jiun.dev${heroImage}` : '',
    authorName: 'Jiun Bae',
    authorUrl: 'https://blog.jiun.dev/about',
    tags: frontmatter.tags
  })

  const breadcrumbSchema = createBreadcrumbSchema([
    { name: 'Home', url: 'https://blog.jiun.dev' },
    { name: 'Posts', url: 'https://blog.jiun.dev/' },
    { name: title, url: `https://blog.jiun.dev${pageUrl}` }
  ])

  return (
    <>
      <Seo ... publishedTime={frontmatter.dateISO} tags={frontmatter.tags} type="article" />
      <JsonLd data={blogPostingSchema} />
      <JsonLd data={breadcrumbSchema} />
    </>
  )
}
```

#### Note (src/views/Note/Note.tsx)
- Article 스키마
- BreadcrumbList 스키마

#### Review (src/views/Review/Review.tsx)
- Review 스키마 (rating, itemReviewed 포함)
- BreadcrumbList 스키마

#### About (src/views/About/About.tsx)
- Person 스키마
- BreadcrumbList 스키마

```tsx
const personSchema = createPersonSchema({
  name: '배지운',
  alternateName: 'Jiun Bae',
  url: 'https://blog.jiun.dev/about',
  image: 'https://blog.jiun.dev/profile.png',
  jobTitle: 'Software Engineer',
  sameAs: [
    'https://github.com/jiunbae',
    'https://linkedin.com/in/jiunbae',
    'https://twitter.com/baejiun'
  ]
})
```

### 3. Notes/Reviews 목록 페이지
**파일**: `src/views/Notes/Notes.tsx`, `src/views/Reviews/Reviews.tsx`

- BreadcrumbList 스키마만 추가
- type="website" 설정

## 완료 조건
- [ ] 모든 템플릿에 dateISO GraphQL 필드 추가됨
- [ ] Home 페이지에 WebSite 스키마 추가됨
- [ ] Post 페이지에 BlogPosting + BreadcrumbList 추가됨
- [ ] Note 페이지에 Article + BreadcrumbList 추가됨
- [ ] Review 페이지에 Review + BreadcrumbList 추가됨
- [ ] About 페이지에 Person + BreadcrumbList 추가됨
- [ ] 모든 Seo 컴포넌트에 type, publishedTime, tags props 전달됨
- [ ] TypeScript 타입 오류 없음
- [ ] 빌드 성공
