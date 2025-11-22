import React from 'react'
import * as styles from './styles.module.scss'

export interface MediaMetadataProps {
  metadata: {
    originalTitle?: string
    year?: number
    director?: string
    creator?: string
    author?: string
    genre?: string[]
    runtime?: string
    pages?: string
  }
  mediaType: 'movie' | 'series' | 'animation' | 'book'
  className?: string
}

const MediaMetadata: React.FC<MediaMetadataProps> = ({ metadata, mediaType, className }) => {
  const metadataItems: Array<{ label: string; value?: string | number | string[] }> = []

  if (metadata.originalTitle) {
    metadataItems.push({ label: '원제', value: metadata.originalTitle })
  }

  if (metadata.year) {
    metadataItems.push({ label: '제작년도', value: metadata.year })
  }

  if (mediaType === 'movie' || mediaType === 'animation') {
    if (metadata.director) {
      metadataItems.push({ label: '감독', value: metadata.director })
    }
  } else if (mediaType === 'series') {
    if (metadata.creator) {
      metadataItems.push({ label: '제작', value: metadata.creator })
    }
  } else if (mediaType === 'book') {
    if (metadata.author) {
      metadataItems.push({ label: '저자', value: metadata.author })
    }
  }

  if (metadata.genre && metadata.genre.length > 0) {
    metadataItems.push({ label: '장르', value: metadata.genre })
  }

  if (metadata.runtime) {
    metadataItems.push({
      label: mediaType === 'book' ? '페이지' : '러닝타임',
      value: metadata.runtime
    })
  }

  if (metadata.pages) {
    metadataItems.push({ label: '페이지', value: metadata.pages })
  }

  if (metadataItems.length === 0) {
    return null
  }

  const renderValue = (value: string | number | string[]) => {
    if (Array.isArray(value)) {
      return value.join(', ')
    }
    return String(value)
  }

  return (
    <div className={`${styles.metadata} ${className || ''}`}>
      {metadataItems.map((item, index) => (
        <div key={index} className={styles.metadataItem}>
          <dt className={styles.label}>{item.label}</dt>
          <dd className={styles.value}>{item.value && renderValue(item.value)}</dd>
        </div>
      ))}
    </div>
  )
}

export default MediaMetadata
