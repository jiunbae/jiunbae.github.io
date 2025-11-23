import { useEffect, useMemo, useState } from 'react'

import { $ } from '@/utils'

import { TAGS } from '../constants'

export const usePostInfiniteScroll = (
  allPosts: Queries.HomeQuery['allMarkdownRemark']['nodes'],
  selectedTag: string,
  totalCount: number
) => {
  const posts = useMemo(
    () => allPosts.filter(({ frontmatter }) => {
      // 프로덕션 환경에서는 published: true인 항목만 표시
      if (process.env.NODE_ENV === 'production' && frontmatter.published === false) {
        return false
      }

      return selectedTag === TAGS.ALL || frontmatter.tags.includes(selectedTag)
    }),
    [allPosts, selectedTag]
  )
  const [displayedItems, setDisplayedItems] = useState(8)
  const visiblePosts = posts.slice(0, displayedItems)

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        if (!entries[0].isIntersecting) return

        if (displayedItems <= totalCount) {
          setDisplayedItems(prev => prev + 8)
        } else {
          observer.disconnect()
        }
      },
      { rootMargin: '400px 0px', threshold: 0 }
    )
    observer.observe($<HTMLElement>('footer'))
    return () => observer.disconnect()
  }, [displayedItems, totalCount])

  return { visiblePosts }
}
