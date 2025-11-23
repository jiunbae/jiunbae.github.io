import type { HeadProps, PageProps } from 'gatsby'
import { useState, useMemo } from 'react'

import { FloatingButton, Seo } from '@/components'
import { getRefinedStringValue } from '@/utils'
import MediaFilter, { MediaType, SortOption } from '@/components/MediaFilter'

import { ReviewList } from './components'
import * as styles from './Reviews.module.scss'

const Reviews = ({ data }: PageProps<Queries.ReviewsQuery>) => {
  const { nodes: allReviews, totalCount, group } = data.allMarkdownRemark

  const [selectedMediaType, setSelectedMediaType] = useState<MediaType>('all')
  const [selectedSort, setSelectedSort] = useState<SortOption>('latest')
  const [selectedTags, setSelectedTags] = useState<string[]>([])

  const allTags = useMemo(() => {
    return (group || [])
      .map(g => g.fieldValue!)
      .filter(Boolean)
      .sort((a, b) => a.localeCompare(b, 'ko'))
  }, [group])

  const handleTagClick = (tag: string) => {
    setSelectedTags(prev =>
      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
    )
  }

  const filteredAndSortedReviews = useMemo(() => {
    let result = [...allReviews]

    // 프로덕션 환경에서는 published: true인 항목만 표시
    if (process.env.NODE_ENV === 'production') {
      result = result.filter(review => review.frontmatter.published !== false)
    }

    if (selectedMediaType !== 'all') {
      result = result.filter(
        review => review.frontmatter.mediaType === selectedMediaType
      )
    }

    if (selectedTags.length > 0) {
      result = result.filter(review =>
        review.frontmatter.tags?.some(tag => selectedTags.includes(tag))
      )
    }

    result.sort((a, b) => {
      switch (selectedSort) {
        case 'latest':
          return (
            new Date(b.frontmatter.sortDate).getTime() -
            new Date(a.frontmatter.sortDate).getTime()
          )
        case 'oldest':
          return (
            new Date(a.frontmatter.sortDate).getTime() -
            new Date(b.frontmatter.sortDate).getTime()
          )
        case 'rating-high':
          return (b.frontmatter.rating || 0) - (a.frontmatter.rating || 0)
        case 'rating-low':
          return (a.frontmatter.rating || 0) - (b.frontmatter.rating || 0)
        default:
          return 0
      }
    })

    return result
  }, [allReviews, selectedMediaType, selectedTags, selectedSort])

  return (
    <main className={styles.main}>
      <div className={styles.container}>
        <div className={styles.header}>
          <h1 className={styles.title}>Reviews</h1>
          <p className={styles.description}>
            영화, 드라마, 애니메이션, 책에 대한 개인적인 리뷰와 감상을 기록합니다.
          </p>
          <div className={styles.count}>
            총 <strong>{filteredAndSortedReviews.length}</strong>개의 리뷰
          </div>
        </div>

        <div className={styles.content}>
          <aside className={styles.sidebar}>
            <MediaFilter
              selectedMediaType={selectedMediaType}
              selectedSort={selectedSort}
              selectedTags={selectedTags}
              availableTags={allTags}
              onMediaTypeChange={setSelectedMediaType}
              onSortChange={setSelectedSort}
              onTagClick={handleTagClick}
            />
          </aside>

          <div className={styles.reviewsSection}>
            <ReviewList reviews={filteredAndSortedReviews} />
          </div>
        </div>
      </div>

      <FloatingButton />
    </main>
  )
}

export const Head = ({ location, data: { site, file } }: HeadProps<Queries.ReviewsQuery>) => {
  const { href } = location as typeof location & { href?: string }
  const pageUrl = href ?? location.pathname
  const seo = {
    title: `${getRefinedStringValue(site?.siteMetadata.title)} | Reviews`,
    description: '영화, 드라마, 애니메이션, 책에 대한 리뷰와 감상',
    heroImage: getRefinedStringValue(file?.publicURL)
  }

  return <Seo title={seo.title} description={seo.description} heroImage={seo.heroImage} pathname={pageUrl} />
}

export default Reviews
