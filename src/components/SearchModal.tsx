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

export default function SearchModal() {
  const [isOpen, setIsOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [activeIndex, setActiveIndex] = useState(0)
  const [searchIndex, setSearchIndex] = useState<SearchItem[]>([])
  const inputRef = useRef<HTMLInputElement>(null)
  const resultsRef = useRef<HTMLDivElement>(null)
  const dialogRef = useRef<HTMLDivElement>(null)
  const fuseRef = useRef<Fuse<SearchItem> | null>(null)

  // Load search index lazily on first open
  const loadIndex = useCallback(async () => {
    if (searchIndex.length > 0) return
    try {
      const res = await fetch('/search-index.json')
      const data: SearchItem[] = await res.json()
      setSearchIndex(data)
      fuseRef.current = new Fuse(data, {
        keys: [
          { name: 'title', weight: 0.4 },
          { name: 'description', weight: 0.3 },
          { name: 'tags', weight: 0.3 },
        ],
        threshold: 0.4,
        includeScore: true,
      })
    } catch { /* silently fail */ }
  }, [searchIndex.length])

  const results = query.trim() && fuseRef.current
    ? fuseRef.current.search(query).slice(0, 10).map(r => r.item)
    : []

  const open = useCallback(() => {
    setIsOpen(true)
    setQuery('')
    setActiveIndex(0)
    loadIndex()
  }, [loadIndex])

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

  // Focus trap
  useEffect(() => {
    if (!isOpen || !dialogRef.current) return
    const dialog = dialogRef.current
    const focusable = dialog.querySelectorAll<HTMLElement>(
      'input, a[href], button, [tabindex]:not([tabindex="-1"])'
    )
    const first = focusable[0]
    const last = focusable[focusable.length - 1]

    const trap = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return
      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault()
          last?.focus()
        }
      } else {
        if (document.activeElement === last) {
          e.preventDefault()
          first?.focus()
        }
      }
    }
    dialog.addEventListener('keydown', trap)
    return () => dialog.removeEventListener('keydown', trap)
  }, [isOpen, results.length])

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
    <div className={styles.overlay} onClick={close} role="presentation">
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-label="Search"
        onClick={e => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <div className={styles.inputWrapper}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
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
            aria-label="Search query"
          />
          <span className={styles.kbd}>ESC</span>
        </div>

        <div className={styles.results} ref={resultsRef} role="listbox">
          {query.trim() && results.length === 0 && (
            <div className={styles.empty}>No results found for &ldquo;{query}&rdquo;</div>
          )}
          {results.map((item, i) => (
            <a
              key={item.slug}
              href={item.slug}
              className={`${styles.resultItem} ${i === activeIndex ? styles.active : ''}`}
              onMouseEnter={() => setActiveIndex(i)}
              role="option"
              aria-selected={i === activeIndex}
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
