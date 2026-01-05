import type { HeadProps, PageProps } from 'gatsby'
import { Link } from 'gatsby'
import { useMemo, useState } from 'react'

import { JsonLd, Seo, createArticleSchema, createBreadcrumbSchema } from '@/components'
import { getRefinedStringValue } from '@/utils'

import { ShareIcon } from '@/components/icons'
import { buildNoteShareUrl, sanitizeNoteSlug } from '../Notes/utils'
import * as styles from './Note.module.scss'

const SITE_URL = 'https://blog.jiun.dev'
const AUTHOR_NAME = 'Jiun Bae'
const AUTHOR_URL = `${SITE_URL}/about`

type LocationState = {
  from?: string;
};

const MAX_SUMMARY_LENGTH = 120

const getPlainText = (html: string) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim()

const getSummary = (description: string | null | undefined, html: string, length = MAX_SUMMARY_LENGTH) => {
  const base = description?.trim() || getPlainText(html)
  return base.length > length ? `${base.slice(0, length).trim()}…` : base
}

const NotePage = ({ data, location }: PageProps<Queries.NoteTemplateQuery, object, LocationState>) => {
  const note = data.markdownRemark

  if (!note) {
    throw new Error('노트 데이터를 찾을 수 없습니다.')
  }

  const { html } = note
  const { title, date, tags, slug, description } = note.frontmatter

  if (!slug) {
    throw new Error('노트 슬러그가 존재하지 않습니다.')
  }

  const refinedHtml = getRefinedStringValue(html)
  const summary = useMemo(() => getSummary(description, refinedHtml), [description, refinedHtml])
  const [isCopied, setIsCopied] = useState(false)
  const shareUrl = buildNoteShareUrl(slug)
  const normalizedSlug = sanitizeNoteSlug(slug)

  const handleCopy = async () => {
    try {
      if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(shareUrl)
      } else {
        const fallback = document.createElement('input')
        fallback.value = shareUrl
        document.body.appendChild(fallback)
        fallback.select()
        document.execCommand('copy')
        document.body.removeChild(fallback)
      }

      setIsCopied(true)
      window.setTimeout(() => setIsCopied(false), 2000)
    } catch {
      window.prompt('아래 링크를 복사해주세요.', shareUrl)
    }
  }

  const fromListPath = location.state?.from ?? '/notes/'

  return (
    <main className={styles.wrapper}>
      <header className={styles.header}>
        <h1 className={styles.title}>{title}</h1>
        <div className={styles.meta}>
          <time>{date}</time>
          {tags?.length ? (
            <ul className={styles.tags}>
              {tags.filter(Boolean).map(tag => (
                <li key={tag as string} className={styles.tag}>{tag}</li>
              ))}
            </ul>
          ) : null}
        </div>
      </header>
      <section className={styles.content} dangerouslySetInnerHTML={{ __html: refinedHtml }} />
      <div className={styles.actions}>
        <button type="button" className={styles.button} onClick={handleCopy} aria-label="노트 링크 복사">
          <ShareIcon width={18} height={18} />
          {isCopied ? '링크 복사됨' : '링크 복사'}
        </button>
        <Link className={styles.button} to={fromListPath} state={{ note: normalizedSlug }}>
          목록으로 돌아가기
        </Link>
      </div>
    </main>
  )
}

export const Head = ({ data, location }: HeadProps<Queries.NoteTemplateQuery>) => {
  const note = data.markdownRemark
  const { href } = location as typeof location & { href?: string }
  const pageUrl = href ?? location.pathname
  const resolvedUrl = pageUrl.startsWith('http') ? pageUrl : `${SITE_URL}${pageUrl}`

  if (!note) return null

  const { frontmatter, html, excerpt } = note
  const summary = getSummary(frontmatter.description, getRefinedStringValue(html ?? ''), 160) || excerpt || frontmatter.title
  const normalizedTags = frontmatter.tags?.filter(Boolean) as string[] | undefined
  const normalizedSlug = sanitizeNoteSlug(frontmatter.slug)
  const heroImage = `/og/notes/${normalizedSlug}.png`
  const heroImageUrl = heroImage ? `${SITE_URL}${heroImage}` : ''

  const articleSchema = createArticleSchema({
    title: frontmatter.title ?? '',
    description: summary ?? '',
    datePublished: frontmatter.dateISO ?? '',
    dateModified: frontmatter.dateISO ?? '',
    url: resolvedUrl,
    image: heroImageUrl,
    authorName: AUTHOR_NAME,
    authorUrl: AUTHOR_URL,
    tags: normalizedTags
  })

  const breadcrumbSchema = createBreadcrumbSchema([
    { name: 'Home', url: SITE_URL },
    { name: 'Notes', url: `${SITE_URL}/notes` },
    { name: frontmatter.title ?? 'Note', url: resolvedUrl }
  ])

  return (
    <>
      <Seo
        title={frontmatter.title}
        description={summary}
        heroImage={heroImage}
        pathname={pageUrl}
        publishedTime={frontmatter.dateISO ?? undefined}
        type="article"
      />
      <JsonLd data={[articleSchema, breadcrumbSchema]} />
    </>
  )
}

export default NotePage
