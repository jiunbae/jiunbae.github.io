import React from 'react'
import * as styles from './styles.module.scss'

interface StarRatingProps {
  rating: number
  maxRating?: number
  size?: 'small' | 'medium' | 'large'
  showNumber?: boolean
  className?: string
}

const StarRating: React.FC<StarRatingProps> = ({
  rating,
  maxRating = 5,
  size = 'medium',
  showNumber = false,
  className
}) => {
  const percentage = (rating / maxRating) * 100

  const renderStars = () => {
    const stars = []
    for (let i = 1; i <= maxRating; i++) {
      const fillPercentage = Math.min(Math.max((rating - (i - 1)) * 100, 0), 100)
      stars.push(
        <span key={i} className={styles.starWrapper}>
          <svg
            className={styles.starEmpty}
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <svg
            className={styles.starFilled}
            style={{ clipPath: `inset(0 ${100 - fillPercentage}% 0 0)` }}
            viewBox="0 0 24 24"
            fill="currentColor"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"
              fill="currentColor"
            />
          </svg>
        </span>
      )
    }
    return stars
  }

  return (
    <div className={`${styles.starRating} ${styles[size]} ${className || ''}`}>
      <div className={styles.stars}>{renderStars()}</div>
      {showNumber && (
        <span className={styles.ratingNumber}>
          {rating.toFixed(1)} / {maxRating}
        </span>
      )}
    </div>
  )
}

export default StarRating
