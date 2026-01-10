import clsx from 'clsx'
import { useState, type KeyboardEvent, type MouseEvent } from 'react'

import { getRefinedStringValue } from '@/utils'

import { TagButtonWithCount } from '../TagButtonWithCount'
import * as styles from './TagList.module.scss'

interface TagListProps {
  tags: { fieldValue: string | null; totalCount: number }[];
  selectedTag: string;
  clickTag: (e: MouseEvent<HTMLElement> | KeyboardEvent<HTMLElement>) => void;
  className?: string;
  maxVisible?: number;
}

export const TagList = ({ tags, selectedTag, clickTag, className, maxVisible = 15 }: TagListProps) => {
  const [isExpanded, setIsExpanded] = useState(false)

  const visibleTags = isExpanded ? tags : tags.slice(0, maxVisible)
  const hasMoreTags = tags.length > maxVisible
  const hiddenCount = tags.length - maxVisible

  return (
    <div className={clsx(styles.tagListWrapper, className)}>
      <ul
        className={styles.tagList}
        onClick={clickTag}
        onKeyDown={clickTag}
        role="presentation"
      >
        {visibleTags.map(({ fieldValue, totalCount }) => {
          const value = getRefinedStringValue(fieldValue)

          return (
            <li key={value} className={styles.tagItem} data-tag={value}>
              <TagButtonWithCount name={value} count={totalCount} isSelected={selectedTag === value} />
            </li>
          )
        })}
      </ul>
      {hasMoreTags && (
        <button
          type="button"
          className={styles.toggleButton}
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {isExpanded ? '접기' : `+${hiddenCount}개 더보기`}
        </button>
      )}
    </div>
  )
}
