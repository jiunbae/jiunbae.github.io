import { useRef, useEffect, useState, useCallback } from 'react'
import type { HeadProps } from 'gatsby'
import clsx from 'clsx'
import { Seo } from '@/components'

import * as styles from './PhysicsSandbox.module.scss'

type PhysicsEngineType = import('./engine').PhysicsEngine
type Tool = 'circle' | 'box' | 'spring' | 'gravity' | 'explosion'

const TOOLS: { key: Tool; label: string; icon: string }[] = [
  { key: 'circle', label: 'Circle', icon: '\u25CB' },
  { key: 'box', label: 'Box', icon: '\u25A1' },
  { key: 'spring', label: 'Spring', icon: '\u223F' },
  { key: 'gravity', label: 'Gravity Gun', icon: '\u2609' },
  { key: 'explosion', label: 'Explosion', icon: '\u2738' },
]

const PhysicsSandboxPage = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const engineRef = useRef<PhysicsEngineType | null>(null)
  const [isSSR, setIsSSR] = useState(true)
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [activeTool, setActiveTool] = useState<Tool>('circle')
  const [gravity, setGravity] = useState(400)
  const [bounciness, setBounciness] = useState(0.5)
  const [count, setCount] = useState(0)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    if (isSSR || !canvasRef.current) return

    let engine: PhysicsEngineType | null = null
    let cancelled = false

    import('./engine').then(({ PhysicsEngine }) => {
      if (cancelled || !canvasRef.current) return
      engine = new PhysicsEngine(canvasRef.current)
      engine.onBodyCountChange((c) => setCount(c))
      engineRef.current = engine
    })

    return () => {
      cancelled = true
      if (engine) {
        engine.onBodyCountChange(null)
        engine.dispose()
        engineRef.current = null
      }
    }
  }, [isSSR])

  // Sync tool
  useEffect(() => {
    if (engineRef.current) engineRef.current.setTool(activeTool)
  }, [activeTool])

  // Sync gravity
  useEffect(() => {
    if (engineRef.current) engineRef.current.setGravity(gravity)
  }, [gravity])

  // Sync restitution
  useEffect(() => {
    if (engineRef.current) engineRef.current.setRestitution(bounciness)
  }, [bounciness])

  const handleClear = useCallback(() => {
    if (engineRef.current) {
      engineRef.current.clear()
      setCount(0)
    }
  }, [])

  if (isSSR) {
    return <div className={styles.page} />
  }

  return (
    <div className={styles.page}>
      <div ref={canvasRef} className={styles.canvas} role="img" aria-label="Physics sandbox" />

      <div className={clsx(styles.hint, { [styles.hidden]: count > 0 })}>
        Select a tool and draw shapes
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
          <span className={styles.controlLabel}>Tool</span>
          <div className={styles.toolButtons}>
            {TOOLS.map(t => (
              <button
                key={t.key}
                className={clsx(styles.toolButton, { [styles.active]: activeTool === t.key })}
                onClick={() => setActiveTool(t.key)}
                title={t.label}
              >
                {t.icon}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Gravity</span>
          <input
            type="range"
            min="0"
            max="1000"
            step="10"
            value={gravity}
            onChange={e => setGravity(parseFloat(e.target.value))}
            className={styles.slider}
          />
        </div>

        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Bounce</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={bounciness}
            onChange={e => setBounciness(parseFloat(e.target.value))}
            className={styles.slider}
          />
        </div>

        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Bodies</span>
          <span className={styles.stat}>{count}</span>
        </div>

        <div className={styles.panelDivider} />

        <div className={styles.controlGroup}>
          <button
            className={clsx(styles.actionButton, styles.clearButton)}
            onClick={handleClear}
          >
            Clear All
          </button>
        </div>
      </div>
    </div>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => (
  <Seo
    title="Physics Sandbox"
    description="Interactive 2D rigid body physics with circles, boxes, springs, gravity gun, and explosions"
    heroImage=""
    pathname={pathname}
  />
)

export default PhysicsSandboxPage
