import clsx from 'clsx'
import { useEffect, useRef, useState } from 'react'

import { getRefinedStringValue } from '@/utils'

import * as styles from './Note.module.scss'
import { ShareIcon } from '@/components/icons'
import { buildNoteShareUrl, getNoteAnchorId, sanitizeNoteSlug } from '../../utils'

type NoteNode = {
  id: string;
  html: string | null;
  frontmatter: {
    title: string;
    date: string;
    tags: readonly (string | null)[] | null;
    slug: string;
  };
};

type NoteProps = {
  note: NoteNode;
  className?: string;
};

const COLLAPSE_LINE_LIMIT = 11

export const Note = ({ note, className }: NoteProps) => {
  const { frontmatter, html } = note
  const { title, date, tags, slug } = frontmatter
  if (!slug) throw new Error('노트 슬러그가 존재하지 않습니다.')
  const refinedHtml = getRefinedStringValue(html)
  const visibleTags = (tags ?? []).filter((tag): tag is string => Boolean(tag))
  const normalizedSlug = sanitizeNoteSlug(slug)
  const anchorId = getNoteAnchorId(slug)
  const contentRef = useRef<HTMLDivElement>(null)
  const [isCollapsed, setIsCollapsed] = useState(true)
  const [isCollapsible, setIsCollapsible] = useState(false)
  const [collapseHeight, setCollapseHeight] = useState<number | null>(null)
  const [isLinkCopied, setIsLinkCopied] = useState(false)

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

  const handleCopyLink = async () => {
    const shareUrl = buildNoteShareUrl(slug)

    try {
      if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(shareUrl)
      } else {
        const fallbackInput = document.createElement('input')
        fallbackInput.value = shareUrl
        document.body.appendChild(fallbackInput)
        fallbackInput.select()
        document.execCommand('copy')
        document.body.removeChild(fallbackInput)
      }

      setIsLinkCopied(true)
      window.setTimeout(() => setIsLinkCopied(false), 2000)
    } catch {
      window.prompt('아래 링크를 복사해주세요.', shareUrl)
    }
  }

  return (
    <li id={anchorId} className={clsx(styles.note, className)} data-note-slug={normalizedSlug}>
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
        <div className={styles.footer}>
          {visibleTags.length > 0 && (
            <ul className={styles.tags}>
              {visibleTags.map(tag => (
                <li key={tag} className={styles.tag}>{tag}</li>
              ))}
            </ul>
          )}
          <div className={styles.actions}>
            <button
              type="button"
              className={clsx(styles.shareButton, { [styles.copied]: isLinkCopied })}
              onClick={handleCopyLink}
              aria-label="노트 링크 복사"
              title={isLinkCopied ? '링크 복사됨' : '노트 링크 복사'}
            >
              <ShareIcon size={20} />
            </button>
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
        </div>
      </article>
    </li>
  )
}
