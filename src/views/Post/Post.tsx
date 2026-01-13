import type { HeadProps, PageProps } from 'gatsby'
import { GatsbyImage } from 'gatsby-plugin-image'

import { Comments, FloatingButton, JsonLd, Seo, createBlogPostingSchema, createBreadcrumbSchema } from '@/components'
import { getRefinedImage, getRefinedStringValue, sanitizePostSlug } from '@/utils'

import { TableOfContents, TagList } from './components'
import * as styles from './Post.module.scss'

const SITE_URL = 'https://jiun.dev'
const AUTHOR_NAME = 'Jiun Bae'
const AUTHOR_URL = `${SITE_URL}/about`

const Post = ({ data }: PageProps<Queries.PostQuery>) => {
  if (!data.markdownRemark) throw new Error('마크다운 데이터가 존재하지 않습니다.')
  const { html, tableOfContents, frontmatter } = data.markdownRemark
  const { title, date, tags, heroImage, heroImageAlt } = frontmatter

  const image = heroImage && getRefinedImage(heroImage?.childImageSharp?.gatsbyImageData)

  return (
    <main className={styles.wrapper}>
      <h1 className={styles.title}>{title}</h1>
      <p className={styles.date}>{date}</p>
      <TagList tags={tags} className={styles.tagList} />
      {image !== null && (
        <GatsbyImage
          image={image}
          alt={heroImageAlt || ''}
          className={styles.heroImage}
          objectFit="contain"
          loading="eager"
        />
      )}
      <div className={styles.contentWrapper}>
        <section className={styles.content} dangerouslySetInnerHTML={{ __html: getRefinedStringValue(html) }} />
        <TableOfContents html={getRefinedStringValue(tableOfContents)} />
      </div>
      <Comments />
      <FloatingButton />
    </main>
  )
}

export const Head = ({ data: { markdownRemark }, location }: HeadProps<Queries.PostQuery>) => {
  const { href } = location as typeof location & { href?: string }
  const pageUrl = href ?? location.pathname
  const resolvedUrl = pageUrl.startsWith('http') ? pageUrl : `${SITE_URL}${pageUrl}`

  if (!markdownRemark) return null

  const { frontmatter, excerpt } = markdownRemark
  const { title, description, dateISO, tags } = frontmatter
  const normalizedTags = tags?.filter(Boolean) as string[] | undefined
  const slug = frontmatter.slug
  const normalizedSlug = slug ? sanitizePostSlug(slug) : null
  const heroImage = normalizedSlug ? `/og/posts/${normalizedSlug}.png` : ''
  const heroImageUrl = heroImage ? `${SITE_URL}${heroImage}` : ''

  const blogPostingSchema = createBlogPostingSchema({
    title: title ?? '',
    description: description ?? excerpt ?? '',
    datePublished: dateISO ?? '',
    dateModified: dateISO ?? '',
    url: resolvedUrl,
    image: heroImageUrl,
    authorName: AUTHOR_NAME,
    authorUrl: AUTHOR_URL,
    tags: normalizedTags
  })

  const breadcrumbSchema = createBreadcrumbSchema([
    { name: 'Home', url: SITE_URL },
    { name: 'Posts', url: `${SITE_URL}/posts` },
    { name: title ?? 'Post', url: resolvedUrl }
  ])

  return (
    <>
      <Seo
        title={title}
        description={description ?? excerpt ?? undefined}
        heroImage={heroImage}
        pathname={pageUrl}
        publishedTime={dateISO ?? undefined}
        tags={normalizedTags}
        type="article"
      />
      <JsonLd data={[blogPostingSchema, breadcrumbSchema]} />
    </>
  )
}

export default Post
