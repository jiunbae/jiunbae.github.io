import { useState, useEffect } from 'react'
import { Link } from 'gatsby'
import clsx from 'clsx'
import { MenuIcon, CloseIcon, SearchIcon, ThemeIcon, ProfileIcon, RssIcon } from '@/components/icons'
import { useTheme } from '@/contexts'

import * as styles from './MobileNav.module.scss'

interface NavLink {
  to: string
  label: string
  state?: { tag?: string }
}

interface MobileNavProps {
  navLinks: NavLink[]
  isActivePath: (path: string) => boolean
  currentPath: string
  onSearchOpen: () => void
}

const getCurrentPageLabel = (path: string): string => {
  if (path.startsWith('/notes')) return 'Notes'
  if (path.startsWith('/reviews')) return 'Reviews'
  if (path.startsWith('/admin')) return 'Admin'
  if (path.startsWith('/about')) return 'About'
  return 'Posts'
}

export const MobileNav = ({ navLinks, isActivePath, currentPath, onSearchOpen }: MobileNavProps) => {
  const [isOpen, setIsOpen] = useState(false)
  const { theme, toggleTheme } = useTheme()

  // 모바일 메뉴 열릴 때 스크롤 방지
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

  // ESC 키로 닫기
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false)
      }
    }

    document.addEventListener('keydown', handleEscape)
    return () => document.removeEventListener('keydown', handleEscape)
  }, [isOpen])

  const handleLinkClick = () => {
    setIsOpen(false)
  }

  const currentLabel = getCurrentPageLabel(currentPath)

  return (
    <>
      {/* 현재 페이지 + Search + 햄버거 버튼 */}
      <div className={styles.mobileNavWrapper}>
        <span className={styles.currentPage}>{currentLabel}</span>
        <button
          className={styles.searchButton}
          onClick={onSearchOpen}
          aria-label="Search"
        >
          <SearchIcon className={styles.icon} />
        </button>
        <button
          className={styles.hamburger}
          onClick={() => setIsOpen(!isOpen)}
          aria-label="Toggle navigation menu"
          aria-expanded={isOpen}
        >
          {isOpen ? (
            <CloseIcon className={styles.icon} />
          ) : (
            <MenuIcon className={styles.icon} />
          )}
        </button>
      </div>

      {/* 오버레이 */}
      {isOpen && (
        <div
          className={styles.overlay}
          onClick={() => setIsOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* 드로어 */}
      <nav
        className={clsx(styles.drawer, { [styles.open]: isOpen })}
        aria-label="Mobile navigation"
      >
        <div className={styles.drawerContent}>
          {/* 메인 네비게이션 링크 */}
          <div className={styles.navLinks}>
            {navLinks.map(link => (
              <Link
                key={link.label}
                to={link.to}
                state={link.state}
                className={clsx(styles.navLink, {
                  [styles.active]: isActivePath(link.to)
                })}
                onClick={handleLinkClick}
                aria-current={isActivePath(link.to) ? "page" : undefined}
              >
                {link.label}
              </Link>
            ))}
          </div>

          {/* 구분선 */}
          <hr className={styles.divider} />

          {/* 액션 버튼들 */}
          <div className={styles.actions}>
            <Link
              to="/about/"
              className={styles.actionButton}
              onClick={handleLinkClick}
            >
              <ProfileIcon className={styles.actionIcon} aria-hidden="true" />
              <span>About</span>
            </Link>

            <Link
              to="/rss.xml"
              className={styles.actionButton}
              onClick={handleLinkClick}
            >
              <RssIcon className={styles.actionIcon} aria-hidden="true" />
              <span>RSS Feed</span>
            </Link>

            <button
              className={styles.actionButton}
              onClick={() => {
                toggleTheme()
                handleLinkClick()
              }}
              aria-label="Toggle theme"
            >
              <ThemeIcon className={styles.actionIcon} aria-hidden="true" />
              <span>Theme</span>
            </button>
          </div>
        </div>
      </nav>
    </>
  )
}
