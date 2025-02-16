import clsx from 'clsx'
import { useState, useEffect, useRef } from 'react'
import { Link, useStaticQuery, graphql } from 'gatsby'

import { useTheme } from '@/contexts'
import ThemeIcon from '@/images/icons/theme.svg'
import RSSIcon from '@/images/icons/rss.svg'
import ProfileIcon from '@/images/icons/profile.svg'
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
              <span className={styles.heading}>{heading}</span>
            </h1>
          </Link>
          <div className={styles.headerButtons}>
            <Link to="/about/">
              <ProfileIcon className={styles.icon} />
            </Link>
            <Link to="/rss.xml">
              <RSSIcon className={styles.icon} />
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
