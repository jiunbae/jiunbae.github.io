import type { HeadProps, PageProps } from 'gatsby'
import { GatsbyImage, getSrc } from 'gatsby-plugin-image'

import { FloatingButton, Seo } from '@/components'
import { getRefinedImage, getRefinedStringValue } from '@/utils'

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
        <GatsbyImage image={image} alt={heroImageAlt || ''} className={styles.heroImage} objectFit="contain" />
      )}
      <div className={styles.contentWrapper}>
        <section className={styles.content} dangerouslySetInnerHTML={{ __html: getRefinedStringValue(html) }} />
        <TableOfContents html={getRefinedStringValue(tableOfContents)} />
      </div>
      <FloatingButton />
    </main>
  )
}

export const Head = ({ data: { markdownRemark }, location: { pathname } }: HeadProps<Queries.PostQuery>) => {
  const seo = {
    title: markdownRemark?.frontmatter.title,
    description: markdownRemark?.frontmatter.description,
    heroImage: markdownRemark?.frontmatter.heroImage
  }

  const image = seo.heroImage && getSrc(getRefinedImage(seo.heroImage?.childImageSharp?.gatsbyImageData))

  return <Seo title={seo.title} description={seo.description} heroImage={image || ''} pathname={pathname} />
}

export default Post
