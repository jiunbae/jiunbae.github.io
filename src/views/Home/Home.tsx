import type { HeadProps, PageProps } from 'gatsby'
import { FloatingButton, Seo } from '@/components'
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

export const Head = ({ location: { pathname }, data: { site, file } }: HeadProps<Queries.HomeQuery>) => {
  const seo = {
    title: site?.siteMetadata.title,
    description: site?.siteMetadata.description,
    heroImage: getRefinedStringValue(file?.publicURL)
  }

  return <Seo title={seo.title} description={seo.description} heroImage={seo.heroImage} pathname={pathname}></Seo>
}

export default Home
