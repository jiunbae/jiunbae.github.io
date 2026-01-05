import type { HeadProps, PageProps } from 'gatsby'
import { FloatingButton, JsonLd, Seo, createWebSiteSchema } from '@/components'
import { getRefinedStringValue } from '@/utils'

import { PostList, TagList } from './components'
import { usePostInfiniteScroll, useTag } from './hooks'
import * as styles from './Home.module.scss'

interface LocationState {
  tag?: string;
}

const Home = ({ data, location }: PageProps<Queries.HomeQuery, object, LocationState>) => {
  const { nodes: allPosts, totalCount, group } = data.allMarkdownRemark
  const { tags, selectedTag, clickTag } = useTag(totalCount, group, location.state?.tag, { pathname: '/' })
  const { visiblePosts } = usePostInfiniteScroll(allPosts, selectedTag, totalCount)

  return (
    <main className={styles.main}>
      <section className={styles.wrapper}>
        <TagList tags={tags} selectedTag={selectedTag} clickTag={clickTag} className={styles.tagList} />
        <PostList posts={visiblePosts} className={styles.postList} />
      </section>
      <FloatingButton />
    </main>
  )
}

export const Head = ({ location, data: { site, file } }: HeadProps<Queries.HomeQuery>) => {
  const { href } = location as typeof location & { href?: string }
  const pageUrl = href ?? location.pathname
  const siteUrl = site?.siteMetadata.siteUrl ?? 'https://blog.jiun.dev'
  const siteName = site?.siteMetadata.title ?? 'Jiunbae\'s Blog'
  const description = site?.siteMetadata.description ?? ''
  const seo = {
    title: site?.siteMetadata.title,
    description: site?.siteMetadata.description,
    heroImage: getRefinedStringValue(file?.publicURL)
  }

  const webSiteSchema = createWebSiteSchema({
    siteUrl,
    siteName,
    description
  })

  return (
    <>
      <Seo title={seo.title} description={seo.description} heroImage={seo.heroImage} pathname={pageUrl} type="website" />
      <JsonLd data={webSiteSchema} />
    </>
  )
}

export default Home
