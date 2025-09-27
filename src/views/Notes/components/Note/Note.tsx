import clsx from 'clsx'
import { useEffect, useRef, useState } from 'react'

import { getRefinedStringValue } from '@/utils'

import * as styles from './Note.module.scss'

type NoteNode = {
  id: string;
  html: string | null;
  frontmatter: {
    title: string;
    date: string;
    tags: readonly (string | null)[] | null;
  };
};

type NoteProps = {
  note: NoteNode;
  className?: string;
};

const COLLAPSE_LINE_LIMIT = 5

export const Note = ({ note, className }: NoteProps) => {
  const { frontmatter, html } = note
  const { title, date, tags } = frontmatter
  const refinedHtml = getRefinedStringValue(html)
  const visibleTags = (tags ?? []).filter((tag): tag is string => Boolean(tag))
  const contentRef = useRef<HTMLDivElement>(null)
  const [isCollapsed, setIsCollapsed] = useState(true)
  const [isCollapsible, setIsCollapsible] = useState(false)
  const [collapseHeight, setCollapseHeight] = useState<number | null>(null)

  useEffect(() => {
    setIsCollapsed(true)
  }, [refinedHtml])

  useEffect(() => {
    const evaluate = () => {
      const element = contentRef.current
      if (!element) return

      const computedStyle = window.getComputedStyle(element)
      const lineHeight = parseFloat(computedStyle.lineHeight || '0')

      if (!Number.isFinite(lineHeight) || lineHeight <= 0) {
        setIsCollapsible(false)
        setCollapseHeight(null)
        setIsCollapsed(false)
        return
      }

      const maxHeight = lineHeight * COLLAPSE_LINE_LIMIT
      const shouldCollapse = element.scrollHeight > maxHeight + 1

      setCollapseHeight(shouldCollapse ? maxHeight : null)
      setIsCollapsible(shouldCollapse)

      if (!shouldCollapse) {
        setIsCollapsed(false)
      }
    }

    evaluate()
    window.addEventListener('resize', evaluate)

    return () => {
      window.removeEventListener('resize', evaluate)
    }
  }, [refinedHtml])

  const contentStyle = isCollapsible && isCollapsed && collapseHeight !== null
    ? { maxHeight: collapseHeight }
    : undefined

  const toggleCollapse = () => {
    setIsCollapsed(prev => !prev)
  }

  return (
    <li className={clsx(styles.note, className)}>
      <article>
        <header className={styles.header}>
          <h2 className={styles.title}>{title}</h2>
          <time className={styles.date}>{date}</time>
        </header>
        <section
          ref={contentRef}
          style={contentStyle}
          className={clsx(styles.content, {
            [styles.collapsible]: isCollapsible,
            [styles.collapsed]: isCollapsible && isCollapsed
          })}
          dangerouslySetInnerHTML={{ __html: refinedHtml }}
        />
        {(visibleTags.length > 0 || isCollapsible) && (
          <div className={styles.footer}>
            {visibleTags.length > 0 && (
              <ul className={styles.tags}>
                {visibleTags.map(tag => (
                  <li key={tag} className={styles.tag}>{tag}</li>
                ))}
              </ul>
            )}
            {isCollapsible && (
              <button
                type="button"
                className={styles.toggleButton}
                onClick={toggleCollapse}
                aria-expanded={!isCollapsed}
              >
                {isCollapsed ? '더보기' : '접기'}
              </button>
            )}
          </div>
        )}
      </article>
    </li>
  )
}
