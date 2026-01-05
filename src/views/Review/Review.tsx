import type { HeadProps, PageProps } from 'gatsby'
import { GatsbyImage, getImage } from 'gatsby-plugin-image'

import { FloatingButton, JsonLd, Seo, createBreadcrumbSchema, createReviewSchema } from '@/components'
import { getRefinedStringValue, sanitizeReviewSlug } from '@/utils'
import StarRating from '@/components/StarRating'
import MediaMetadata from '@/components/MediaMetadata'
import { Tag } from '@/components/Tag'

import * as styles from './Review.module.scss'

const SITE_URL = 'https://blog.jiun.dev'
const AUTHOR_NAME = 'Jiun Bae'
const AUTHOR_URL = `${SITE_URL}/about`

const mediaTypeLabels: Record<string, string> = {
  movie: '영화',
  series: '드라마',
  animation: '애니메이션',
  book: '책'
}

const Review = ({ data }: PageProps<Queries.ReviewQuery>) => {
  if (!data.markdownRemark) throw new Error('마크다운 데이터가 존재하지 않습니다.')
  const { html, frontmatter } = data.markdownRemark
  const { title, date, mediaType, rating, oneLiner, tags, poster, metadata } = frontmatter

  const posterImage = poster?.childImageSharp?.gatsbyImageData
    ? getImage(poster.childImageSharp.gatsbyImageData)
    : undefined

  return (
    <main className={styles.wrapper}>
      <div className={styles.reviewHeader}>
        <div className={styles.posterSection}>
          {posterImage ? (
            <GatsbyImage
              image={posterImage}
              alt={title}
              className={styles.poster}
              loading="eager"
            />
          ) : (
            <div className={styles.posterPlaceholder}>No Image</div>
          )}
        </div>

        <div className={styles.headerContent}>
          <div className={styles.mediaTypeBadge}>{mediaType ? mediaTypeLabels[mediaType] : ''}</div>
          <h1 className={styles.title}>{title}</h1>

          {rating !== undefined && rating !== null && (
            <div className={styles.ratingSection}>
              <StarRating rating={rating} size="large" showNumber />
            </div>
          )}

          {oneLiner && <p className={styles.oneLiner}>{oneLiner}</p>}

          <div className={styles.date}>{date}</div>

          {tags && tags.length > 0 && (
            <div className={styles.tags}>
              {tags.map(tag => (
                <Tag key={tag} name={tag} />
              ))}
            </div>
          )}
        </div>
      </div>

      {metadata && mediaType && (
        <MediaMetadata
          metadata={{
            originalTitle: metadata.originalTitle ?? undefined,
            year: metadata.year ?? undefined,
            director: metadata.director ?? undefined,
            creator: metadata.creator ?? undefined,
            author: metadata.author ?? undefined,
            genre: metadata.genre ? [...metadata.genre] : undefined,
            runtime: metadata.runtime ?? undefined,
            pages: metadata.pages ?? undefined,
          }}
          mediaType={mediaType as 'movie' | 'series' | 'animation' | 'book'}
          className={styles.metadata}
        />
      )}

      <section className={styles.content} dangerouslySetInnerHTML={{ __html: getRefinedStringValue(html) }} />

      <FloatingButton />
    </main>
  )
}

export const Head = ({ data: { markdownRemark }, location }: HeadProps<Queries.ReviewQuery>) => {
  const { href } = location as typeof location & { href?: string }
  const pageUrl = href ?? location.pathname
  const resolvedUrl = pageUrl.startsWith('http') ? pageUrl : `${SITE_URL}${pageUrl}`

  if (!markdownRemark) return null

  const { frontmatter, excerpt } = markdownRemark
  const slug = frontmatter.slug
  const normalizedSlug = slug ? sanitizeReviewSlug(slug) : null
  const heroImage = normalizedSlug ? `/og/reviews/${normalizedSlug}.png` : ''
  const heroImageUrl = heroImage ? `${SITE_URL}${heroImage}` : ''

  const creatorName = frontmatter.mediaType === 'book'
    ? frontmatter.metadata?.author ?? undefined
    : frontmatter.metadata?.director
      ?? frontmatter.metadata?.creator
      ?? frontmatter.metadata?.author
      ?? undefined

  const reviewSchema = createReviewSchema({
    title: frontmatter.title ?? '',
    description: frontmatter.oneLiner ?? excerpt ?? '',
    datePublished: frontmatter.dateISO ?? '',
    url: resolvedUrl,
    image: heroImageUrl,
    rating: frontmatter.rating ?? 0,
    mediaType: frontmatter.mediaType as 'movie' | 'series' | 'animation' | 'book',
    authorName: AUTHOR_NAME,
    authorUrl: AUTHOR_URL,
    itemReviewed: {
      name: frontmatter.metadata?.originalTitle ?? frontmatter.title ?? '',
      url: resolvedUrl,
      image: heroImageUrl,
      creatorName
    }
  })

  const breadcrumbSchema = createBreadcrumbSchema([
    { name: 'Home', url: SITE_URL },
    { name: 'Reviews', url: `${SITE_URL}/reviews` },
    { name: frontmatter.title ?? 'Review', url: resolvedUrl }
  ])

  return (
    <>
      <Seo
        title={frontmatter.title}
        description={frontmatter.oneLiner ?? excerpt ?? undefined}
        heroImage={heroImage}
        pathname={pageUrl}
        publishedTime={frontmatter.dateISO ?? undefined}
        type="article"
      />
      <JsonLd data={[reviewSchema, breadcrumbSchema]} />
    </>
  )
}

export default Review
