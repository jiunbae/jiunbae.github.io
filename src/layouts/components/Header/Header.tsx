import clsx from 'clsx'
import { useState, useEffect, useRef } from 'react'
import { Link, useStaticQuery, graphql } from 'gatsby'

import { useTheme } from '@/contexts'
import { ThemeIcon, RssIcon, ProfileIcon, LogoIcon } from '@/components/icons'
import { reactCss } from '@/utils'

import * as styles from './Header.module.scss'
import { useScrollIndicator } from './hooks'

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
  const { theme, toggleTheme } = useTheme()
  const { isPost, progressWidth } = useScrollIndicator(pathname)
  const headerRef = useRef<HTMLElement>(null)

  const isAdminPage = pathname.startsWith('/admin')

  const baseNavLinks = [
    { to: '/', label: 'Posts', state: { tag: undefined } },
    { to: '/notes/', label: 'Notes', state: { tag: undefined } },
    { to: '/reviews/', label: 'Reviews', state: { tag: undefined } }
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
    const handleScroll = () => {
      setIsShrink(window.scrollY > 0)
    }

    window.addEventListener('scroll', handleScroll)
    
    return () => {
      window.removeEventListener('scroll', handleScroll)
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
            <nav className={styles.navigation}>
              {navLinks.map(link => (
                <Link
                  key={link.label}
                  to={link.to}
                  state={link.state}
                  className={clsx(styles.link, { [styles.activeLink]: isActivePath(link.to) })}
                >
                  {link.label}
                </Link>
              ))}
            </nav>
            <Link to="/about/">
              <ProfileIcon className={styles.icon} />
            </Link>
            <Link to="/rss.xml">
              <RssIcon className={styles.icon} />
            </Link>
            <button
              className={styles.iconButton}
              onClick={toggleTheme}
              tabIndex={0}
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              <ThemeIcon className={clsx(styles.icon, styles.iconFill)} />
            </button>
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
    </>
  )
}
