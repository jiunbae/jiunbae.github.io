import React from 'react'
import { Link } from 'gatsby'
import { GatsbyImage, IGatsbyImageData } from 'gatsby-plugin-image'
import StarRating from '../StarRating'
import * as styles from './styles.module.scss'

export interface MediaCardProps {
  title: string
  slug: string
  mediaType: 'movie' | 'series' | 'animation' | 'book'
  rating?: number
  oneLiner?: string
  poster?: IGatsbyImageData
  year?: number
  genres?: string[]
  className?: string
}

const mediaTypeLabels: Record<string, string> = {
  movie: '영화',
  series: '드라마',
  animation: '애니메이션',
  book: '책'
}

const MediaCard: React.FC<MediaCardProps> = ({
  title,
  slug,
  mediaType,
  rating,
  oneLiner,
  poster,
  year,
  genres = [],
  className
}) => {
  return (
    <Link to={slug} className={`${styles.mediaCard} ${className || ''}`}>
      <div className={styles.posterWrapper}>
        {poster ? (
          <GatsbyImage image={poster} alt={title} className={styles.poster} />
        ) : (
          <div className={styles.posterPlaceholder}>
            <span>No Image</span>
          </div>
        )}
        <div className={styles.badge}>{mediaTypeLabels[mediaType]}</div>
      </div>

      <div className={styles.content}>
        <h3 className={styles.title}>{title}</h3>

        {rating !== undefined && (
          <div className={styles.rating}>
            <StarRating rating={rating} size="small" />
          </div>
        )}

        {oneLiner && <p className={styles.oneLiner}>{oneLiner}</p>}

        {(year || genres.length > 0) && (
          <div className={styles.meta}>
            {year && <span className={styles.year}>{year}</span>}
            {genres.length > 0 && (
              <>
                {year && <span className={styles.separator}>•</span>}
                <span className={styles.genres}>{genres.slice(0, 2).join(', ')}</span>
              </>
            )}
          </div>
        )}
      </div>
    </Link>
  )
}

export default MediaCard
