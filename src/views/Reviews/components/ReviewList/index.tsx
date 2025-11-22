import React from 'react'
import { getImage } from 'gatsby-plugin-image'
import MediaCard from '@/components/MediaCard'
import * as styles from './styles.module.scss'

interface ReviewListProps {
  reviews: readonly any[]
  className?: string
}

export const ReviewList: React.FC<ReviewListProps> = ({ reviews, className }) => {
  if (reviews.length === 0) {
    return (
      <div className={styles.emptyState}>
        <p>표시할 리뷰가 없습니다.</p>
      </div>
    )
  }

  return (
    <div className={`${styles.reviewList} ${className || ''}`}>
      {reviews.map(review => {
        const posterImage = review.frontmatter.poster?.childImageSharp?.gatsbyImageData
          ? getImage(review.frontmatter.poster.childImageSharp.gatsbyImageData)
          : undefined

        return (
          <MediaCard
            key={review.id}
            title={review.frontmatter.title}
            slug={review.frontmatter.slug}
            mediaType={review.frontmatter.mediaType as 'movie' | 'series' | 'animation' | 'book'}
            rating={review.frontmatter.rating ?? undefined}
            oneLiner={review.frontmatter.oneLiner ?? undefined}
            poster={posterImage}
            year={review.frontmatter.metadata?.year ?? undefined}
            genres={review.frontmatter.metadata?.genre ?? []}
          />
        )
      })}
    </div>
  )
}
