import { type PropsWithChildren, useState, useEffect, useCallback, useRef } from 'react'
import { navigate } from 'gatsby'
import clsx from 'clsx'

import { Footer, Header, SkipLink } from './components'
import playgroundItems from '@/data/playground'
import * as styles from './Layout.module.scss'

type LayoutProps = PropsWithChildren<{ pathname: string }>;

const immersivePaths = new Set(playgroundItems.map(item => item.slug))

const isImmersivePage = (pathname: string) => {
  const normalized = pathname.endsWith('/') ? pathname : `${pathname}/`
  return immersivePaths.has(normalized)
}

const ImmersiveLayout = ({ pathname, children }: LayoutProps) => {
  const [headerVisible, setHeaderVisible] = useState(false)
  const [hintDone, setHintDone] = useState(false)
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const showHeader = useCallback(() => {
    if (hideTimer.current) clearTimeout(hideTimer.current)
    setHeaderVisible(true)
    setHintDone(true)
  }, [])

  const hideHeader = useCallback(() => {
    hideTimer.current = setTimeout(() => setHeaderVisible(false), 400)
  }, [])

  // Hide hint after animation finishes (4s)
  useEffect(() => {
    const timer = setTimeout(() => setHintDone(true), 4000)
    return () => clearTimeout(timer)
  }, [])

  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        navigate('/playground/')
      }
    }
    window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [])

  useEffect(() => {
    // Touch: swipe down from top edge
    let touchStartY = 0
    const handleTouchStart = (e: TouchEvent) => {
      touchStartY = e.touches[0].clientY
    }
    const handleTouchMove = (e: TouchEvent) => {
      if (touchStartY < 30 && e.touches[0].clientY - touchStartY > 40) {
        showHeader()
      }
    }
    window.addEventListener('touchstart', handleTouchStart, { passive: true })
    window.addEventListener('touchmove', handleTouchMove, { passive: true })
    return () => {
      window.removeEventListener('touchstart', handleTouchStart)
      window.removeEventListener('touchmove', handleTouchMove)
    }
  }, [showHeader])

  return (
    <div className={styles.immersiveWrapper}>
      {/* Visual hint: glowing line + label */}
      <div className={clsx(styles.topHint, { [styles.hintHidden]: hintDone })} />
      <div className={clsx(styles.topHintLabel, { [styles.hintHidden]: hintDone })}>
        hover here for navigation
      </div>

      {/* Invisible hover trigger at top edge */}
      <div
        className={styles.hoverZone}
        onMouseEnter={showHeader}
      />

      {/* Slide-down header overlay */}
      <div
        className={clsx(styles.headerOverlay, { [styles.visible]: headerVisible })}
        onMouseEnter={showHeader}
        onMouseLeave={hideHeader}
      >
        <Header pathname={pathname} />
      </div>

      {children}
    </div>
  )
}

const Layout = ({ pathname, children }: LayoutProps) => {
  if (isImmersivePage(pathname)) {
    return <ImmersiveLayout pathname={pathname}>{children}</ImmersiveLayout>
  }

  return (
    <>
      <SkipLink />
      <Header pathname={pathname} />
      <main id="main-content">
        {children}
      </main>
      <Footer />
    </>
  )
}

export default Layout
