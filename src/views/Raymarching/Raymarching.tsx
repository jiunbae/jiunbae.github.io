import { useRef, useEffect, useState, useCallback } from 'react'
import type { HeadProps } from 'gatsby'
import clsx from 'clsx'
import { Seo } from '@/components'

import * as styles from './Raymarching.module.scss'
import type { RaymarchEngine } from './engine'

const SCENE_NAMES = [
  'Alien Planet',
  'Crystal Cave',
  'Fractal Landscape',
  'Abstract Geometry'
]

const RaymarchingPage = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const engineRef = useRef<RaymarchEngine | null>(null)
  const [isSSR, setIsSSR] = useState(true)
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [activeScene, setActiveScene] = useState(0)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    if (isSSR || !canvasRef.current) return

    let engine: RaymarchEngine | null = null
    let cancelled = false

    import('./engine').then(({ RaymarchEngine }) => {
      if (cancelled || !canvasRef.current) return
      engine = new RaymarchEngine(canvasRef.current)
      engineRef.current = engine
    })

    return () => {
      cancelled = true
      if (engine) {
        engine.dispose()
        engineRef.current = null
      }
    }
  }, [isSSR])

  useEffect(() => {
    if (engineRef.current) {
      engineRef.current.setScene(activeScene)
    }
  }, [activeScene])

  const handleSceneChange = useCallback((index: number) => {
    setActiveScene(index)
  }, [])

  if (isSSR) {
    return <div className={styles.page} />
  }

  return (
    <div className={styles.page}>
      <div ref={canvasRef} className={styles.canvas} />

      <div className={styles.hint}>
        Move mouse to look around
      </div>

      <button
        className={styles.panelToggle}
        onClick={() => setIsPanelOpen(prev => !prev)}
        aria-label={isPanelOpen ? 'Hide controls' : 'Show controls'}
      >
        {isPanelOpen ? '\u2715' : '\u2699'}
      </button>

      <div className={clsx(styles.panel, { [styles.panelClosed]: !isPanelOpen })}>
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>World</span>
        </div>

        <div className={styles.sceneButtons}>
          {SCENE_NAMES.map((name, i) => (
            <button
              key={name}
              className={clsx(styles.sceneButton, { [styles.active]: activeScene === i })}
              onClick={() => handleSceneChange(i)}
            >
              {name}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => (
  <Seo
    title="Raymarching Worlds"
    description="Real-time raymarched 3D worlds rendered with pure WebGL fragment shaders -- alien planets, crystal caves, fractal landscapes"
    heroImage=""
    pathname={pathname}
  />
)

export default RaymarchingPage
