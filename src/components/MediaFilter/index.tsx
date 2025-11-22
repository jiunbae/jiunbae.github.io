import React from 'react'
import clsx from 'clsx'
import * as styles from './styles.module.scss'

export type MediaType = 'all' | 'movie' | 'series' | 'animation' | 'book'
export type SortOption = 'latest' | 'oldest' | 'rating-high' | 'rating-low'

interface MediaFilterProps {
  selectedMediaType: MediaType
  selectedSort: SortOption
  selectedTags: string[]
  availableTags: string[]
  onMediaTypeChange: (mediaType: MediaType) => void
  onSortChange: (sort: SortOption) => void
  onTagClick: (tag: string) => void
  className?: string
}

const mediaTypeOptions: Array<{ value: MediaType; label: string }> = [
  { value: 'all', label: '전체' },
  { value: 'movie', label: '영화' },
  { value: 'series', label: '드라마' },
  { value: 'animation', label: '애니메이션' },
  { value: 'book', label: '책' }
]

const sortOptions: Array<{ value: SortOption; label: string }> = [
  { value: 'latest', label: '최신순' },
  { value: 'oldest', label: '오래된 순' },
  { value: 'rating-high', label: '높은 별점순' },
  { value: 'rating-low', label: '낮은 별점순' }
]

const MediaFilter: React.FC<MediaFilterProps> = ({
  selectedMediaType,
  selectedSort,
  selectedTags,
  availableTags,
  onMediaTypeChange,
  onSortChange,
  onTagClick,
  className
}) => {
  return (
    <div className={clsx(styles.mediaFilter, className)}>
      <div className={styles.filterSection}>
        <h3 className={styles.filterTitle}>미디어 타입</h3>
        <div className={styles.filterOptions}>
          {mediaTypeOptions.map(option => (
            <button
              key={option.value}
              className={clsx(
                styles.filterButton,
                selectedMediaType === option.value && styles.active
              )}
              onClick={() => onMediaTypeChange(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.filterSection}>
        <h3 className={styles.filterTitle}>정렬</h3>
        <div className={styles.filterOptions}>
          {sortOptions.map(option => (
            <button
              key={option.value}
              className={clsx(
                styles.filterButton,
                selectedSort === option.value && styles.active
              )}
              onClick={() => onSortChange(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      {availableTags.length > 0 && (
        <div className={styles.filterSection}>
          <h3 className={styles.filterTitle}>태그</h3>
          <div className={styles.tagList}>
            {availableTags.map(tag => (
              <button
                key={tag}
                className={clsx(
                  styles.tagButton,
                  selectedTags.includes(tag) && styles.active
                )}
                onClick={() => onTagClick(tag)}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default MediaFilter
