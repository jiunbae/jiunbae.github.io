import { useState, useEffect, useRef, useCallback } from 'react'
import Fuse from 'fuse.js'
import * as styles from './SearchModal.module.scss'

export interface SearchItem {
  title: string
  description: string
  tags: string[]
  slug: string
  type: 'post' | 'note'
}

interface Props {
  searchIndex: SearchItem[]
}

export default function SearchModal({ searchIndex }: Props) {
  const [isOpen, setIsOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [activeIndex, setActiveIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const resultsRef = useRef<HTMLDivElement>(null)

  const fuse = useRef(
    new Fuse(searchIndex, {
      keys: [
        { name: 'title', weight: 0.4 },
        { name: 'description', weight: 0.3 },
        { name: 'tags', weight: 0.3 },
      ],
      threshold: 0.4,
      includeScore: true,
    })
  )

  const results = query.trim()
    ? fuse.current.search(query).slice(0, 10).map(r => r.item)
    : []

  const open = useCallback(() => {
    setIsOpen(true)
    setQuery('')
    setActiveIndex(0)
  }, [])

  const close = useCallback(() => {
    setIsOpen(false)
    setQuery('')
  }, [])

  useEffect(() => {
    const handleToggle = () => open()
    window.addEventListener('toggle-search', handleToggle)
    return () => window.removeEventListener('toggle-search', handleToggle)
  }, [open])

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        isOpen ? close() : open()
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [isOpen, open, close])

  useEffect(() => {
    if (isOpen) {
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [isOpen])

  useEffect(() => {
    setActiveIndex(0)
  }, [query])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      close()
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIndex(i => Math.min(i + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIndex(i => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' && results[activeIndex]) {
      window.location.href = results[activeIndex].slug
    }
  }

  useEffect(() => {
    const el = resultsRef.current?.children[activeIndex] as HTMLElement | undefined
    el?.scrollIntoView({ block: 'nearest' })
  }, [activeIndex])

  if (!isOpen) return null

  return (
    <div className={styles.overlay} onClick={close}>
      <div className={styles.dialog} onClick={e => e.stopPropagation()} onKeyDown={handleKeyDown}>
        <div className={styles.inputWrapper}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.35-4.35" />
          </svg>
          <input
            ref={inputRef}
            className={styles.input}
            type="text"
            placeholder="Search posts and notes..."
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
          <span className={styles.kbd}>ESC</span>
        </div>

        <div className={styles.results} ref={resultsRef}>
          {query.trim() && results.length === 0 && (
            <div className={styles.empty}>No results found for &ldquo;{query}&rdquo;</div>
          )}
          {results.map((item, i) => (
            <a
              key={item.slug}
              href={item.slug}
              className={`${styles.resultItem} ${i === activeIndex ? styles.active : ''}`}
              onMouseEnter={() => setActiveIndex(i)}
            >
              <div className={styles.resultTitle}>
                {item.title}
                <span className={styles.badge}>{item.type}</span>
              </div>
              {item.description && (
                <div className={styles.resultDesc}>{item.description}</div>
              )}
            </a>
          ))}
        </div>

        {results.length > 0 && (
          <div className={styles.footer}>
            <span><span className={styles.kbd}>&uarr;&darr;</span> navigate</span>
            <span><span className={styles.kbd}>&crarr;</span> open</span>
            <span><span className={styles.kbd}>esc</span> close</span>
          </div>
        )}
      </div>
    </div>
  )
}
