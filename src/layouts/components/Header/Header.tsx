import clsx from 'clsx'
import { useState, useEffect, useRef } from 'react'
import { Link, useStaticQuery, graphql } from 'gatsby'

import { useTheme } from '@/contexts'
import { ThemeIcon, RssIcon, ProfileIcon, LogoIcon, SearchIcon } from '@/components/icons'
import { SearchModal } from '@/components/Search'
import { reactCss } from '@/utils'
import { rafThrottle } from '@/utils/performance'

import * as styles from './Header.module.scss'
import { useScrollIndicator } from './hooks'
import { MobileNav } from './components/MobileNav'

interface HeaderProps {
  pathname: string;
};

export const Header = ({ pathname }: HeaderProps) => {
  const { site: { siteMetadata: { heading } } } = useStaticQuery(graphql`
    query {
      site {
        siteMetadata {
          heading
        }
      }
    }
  `)

  const [ isShrink, setIsShrink ] = useState(false)
  const [ isHeaderVisible, setIsHeaderVisible ] = useState(true)
  const [ isSearchOpen, setIsSearchOpen ] = useState(false)
  const { theme, toggleTheme } = useTheme()
  const { isPost, progressWidth } = useScrollIndicator(pathname)
  const headerRef = useRef<HTMLElement>(null)

  const isAdminPage = pathname.startsWith('/admin')

  const baseNavLinks = [
    { to: '/', label: 'Posts', state: { tag: undefined } },
    { to: '/notes/', label: 'Notes', state: { tag: undefined } },
    { to: '/reviews/', label: 'Reviews', state: { tag: undefined } },
    { to: '/playground/', label: 'Playground', state: undefined }
  ]

  const navLinks = isAdminPage
    ? [...baseNavLinks, { to: '/admin/', label: 'Admin', state: undefined }]
    : baseNavLinks

  const normalizePathname = (path: string) => (path.endsWith('/') ? path : `${path}/`)
  const activePathname = normalizePathname(pathname)
  const isActivePath = (target: string) => {
    if (target === '/') {
      return pathname === '/' || pathname.startsWith('/posts/')
    }

    if (target === '/reviews/') {
      return pathname.startsWith('/reviews/')
    }

    if (target === '/admin/') {
      return pathname.startsWith('/admin')
    }

    if (target === '/playground/') {
      return pathname.startsWith('/playground')
    }

    if (target === '/about/') {
      return pathname.startsWith('/about')
    }

    return activePathname === normalizePathname(target)
  }

  useEffect(() => {
    if (!headerRef.current) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsHeaderVisible(entry.isIntersecting)
      },
      { threshold: 0.1 }
    )

    observer.observe(headerRef.current)

    return () => {
      observer.disconnect()
    }
  }, [])

  useEffect(() => {
    const handleScroll = rafThrottle(() => {
      setIsShrink(window.scrollY > 0)
    })

    window.addEventListener('scroll', handleScroll, { passive: true })

    return () => {
      window.removeEventListener('scroll', handleScroll)
    }
  }, [])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setIsSearchOpen(true)
      }
    }

    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  return (
    <>
      <header ref={headerRef} className={clsx(styles.header, styles.fixed, { [styles.shrink]: isShrink })}>
        <div className={styles.wrapper}>
          <Link to="/" state={{ tag: undefined }} className={styles.headingLink}>
            <h1 className={styles.headingWrapper}>
              <LogoIcon className={styles.logoIcon} />
              <span className={styles.heading}>{heading}</span>
            </h1>
          </Link>
          <div className={styles.headerButtons}>
            {/* 데스크톱 네비게이션 */}
            <nav className={clsx(styles.navigation, styles.desktopOnly)} aria-label="Main navigation">
              {navLinks.map(link => (
                <Link
                  key={link.label}
                  to={link.to}
                  state={link.state}
                  className={clsx(styles.link, { [styles.activeLink]: isActivePath(link.to) })}
                  aria-current={isActivePath(link.to) ? "page" : undefined}
                >
                  {link.label}
                </Link>
              ))}
              <Link
                to="/about/"
                aria-label="About page"
                className={clsx(styles.iconLink, styles.desktopOnly, { [styles.activeIconLink]: isActivePath('/about/') })}
              >
                <ProfileIcon className={styles.icon} aria-hidden="true" />
              </Link>
              <Link to="/rss.xml" aria-label="RSS feed" className={clsx(styles.iconLink, styles.desktopOnly)}>
                <RssIcon className={styles.icon} aria-hidden="true" />
              </Link>
              <button
                className={clsx(styles.iconButton, styles.desktopOnly)}
                onClick={() => setIsSearchOpen(true)}
                tabIndex={0}
                aria-label="Search"
              >
                <SearchIcon className={styles.icon} aria-hidden="true" />
              </button>
              <button
                className={clsx(styles.iconButton, styles.desktopOnly)}
                onClick={toggleTheme}
                aria-label="Toggle theme"
                title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
              >
                <ThemeIcon className={styles.icon} aria-hidden="true" />
              </button>
            </nav>

            {/* 모바일 네비게이션 */}
            <div className={styles.mobileOnly}>
              <MobileNav navLinks={navLinks} isActivePath={isActivePath} currentPath={pathname} onSearchOpen={() => setIsSearchOpen(true)} />
            </div>
          </div>
        </div>
        {isPost && (
          <div
            style={reactCss({ '--progress-width': `${progressWidth}%` })}
            className={clsx(styles.progressBar, {
              [styles.fixedIndicator]: !isHeaderVisible
            })}
          />
        )}
      </header>
      <SearchModal isOpen={isSearchOpen} onClose={() => setIsSearchOpen(false)} />
    </>
  )
}
