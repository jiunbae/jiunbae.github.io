import type { HeadProps, PageProps } from 'gatsby'
import { GatsbyImage } from 'gatsby-plugin-image'

import { Comments, FloatingButton, Seo } from '@/components'
import { getRefinedImage, getRefinedStringValue, sanitizePostSlug } from '@/utils'

import { TableOfContents, TagList } from './components'
import * as styles from './Post.module.scss'

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

  const slug = markdownRemark?.frontmatter.slug
  const normalizedSlug = slug ? sanitizePostSlug(slug) : null
  const heroImage = normalizedSlug ? `/og/posts/${normalizedSlug}.png` : ''

  return (
    <Seo
      title={markdownRemark?.frontmatter.title}
      description={markdownRemark?.frontmatter.description ?? markdownRemark?.excerpt ?? undefined}
      heroImage={heroImage}
      pathname={pageUrl}
    />
  )
}

export default Post
