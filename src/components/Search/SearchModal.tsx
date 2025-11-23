import { useState, useEffect, useRef } from 'react'
import { navigate } from 'gatsby'
import clsx from 'clsx'
import Fuse from 'fuse.js'
import { CloseIcon } from '@/components/icons'
import { useSearchIndex } from './useSearchIndex'

import * as styles from './SearchModal.module.scss'

interface SearchModalProps {
  isOpen: boolean
  onClose: () => void
}

export const SearchModal = ({ isOpen, onClose }: SearchModalProps) => {
  const [query, setQuery] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const resultsRef = useRef<HTMLDivElement>(null)

  const searchIndex = useSearchIndex()

  const fuse = useRef<Fuse<any> | null>(null)

  useEffect(() => {
    if (searchIndex.length > 0) {
      fuse.current = new Fuse(searchIndex, {
        keys: ['title', 'excerpt', 'tags'],
        threshold: 0.3,
        includeScore: true,
      })
    }
  }, [searchIndex])

  const results = query.length > 0 && fuse.current
    ? fuse.current.search(query).slice(0, 10)
    : []

  useEffect(() => {
    if (isOpen) {
      setQuery('')
      setSelectedIndex(0)
      inputRef.current?.focus()
    }
  }, [isOpen])

  useEffect(() => {
    setSelectedIndex(0)
  }, [query])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return

      switch (e.key) {
        case 'Escape':
          onClose()
          break
        case 'ArrowDown':
          e.preventDefault()
          setSelectedIndex(i => Math.min(i + 1, results.length - 1))
          break
        case 'ArrowUp':
          e.preventDefault()
          setSelectedIndex(i => Math.max(i - 1, 0))
          break
        case 'Enter':
          e.preventDefault()
          if (results[selectedIndex]) {
            navigate(results[selectedIndex].item.slug)
            onClose()
          }
          break
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, selectedIndex, results, onClose])

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = 'unset'
    }

    return () => {
      document.body.style.overflow = 'unset'
    }
  }, [isOpen])

  if (!isOpen) return null

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <div className={styles.header}>
          <input
            ref={inputRef}
            type="text"
            className={styles.input}
            placeholder="Search posts, notes, and reviews..."
            value={query}
            onChange={e => setQuery(e.target.value)}
            aria-label="Search"
          />
          <button
            className={styles.closeButton}
            onClick={onClose}
            aria-label="Close search"
          >
            <CloseIcon className={styles.closeIcon} />
          </button>
        </div>

        <div ref={resultsRef} className={styles.results}>
          {query.length === 0 && (
            <div className={styles.emptyState}>
              <p>Start typing to search...</p>
              <div className={styles.hint}>
                <kbd>↑</kbd> <kbd>↓</kbd> to navigate
                <kbd>Enter</kbd> to select
                <kbd>Esc</kbd> to close
              </div>
            </div>
          )}

          {query.length > 0 && results.length === 0 && (
            <div className={styles.emptyState}>
              <p>No results found for "{query}"</p>
            </div>
          )}

          {results.length > 0 && (
            <ul className={styles.resultsList}>
              {results.map((result, index) => (
                <li
                  key={result.item.slug}
                  className={clsx(styles.resultItem, {
                    [styles.selected]: index === selectedIndex
                  })}
                >
                  <a
                    href={result.item.slug}
                    onClick={e => {
                      e.preventDefault()
                      navigate(result.item.slug)
                      onClose()
                    }}
                    className={styles.resultLink}
                  >
                    <div className={styles.resultContent}>
                      <div className={styles.resultTitle}>{result.item.title}</div>
                      {result.item.excerpt && (
                        <div className={styles.resultExcerpt}>{result.item.excerpt}</div>
                      )}
                      <div className={styles.resultMeta}>
                        <span className={styles.resultType}>{result.item.type}</span>
                        {result.item.date && (
                          <span className={styles.resultDate}>{result.item.date}</span>
                        )}
                      </div>
                    </div>
                  </a>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}
