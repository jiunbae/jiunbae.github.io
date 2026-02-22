import { useRef, useEffect, useState, useCallback } from 'react'
import type { HeadProps } from 'gatsby'
import clsx from 'clsx'
import { Seo } from '@/components'

import * as styles from './ShaderArt.module.scss'

type ShaderArtEngine = import('./engine').ShaderArtEngine

const ShaderArtPage = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const engineRef = useRef<ShaderArtEngine | null>(null)
  const [isSSR, setIsSSR] = useState(true)
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [currentName, setCurrentName] = useState('')
  const [currentIndex, setCurrentIndex] = useState(0)
  const [shaderCount, setShaderCount] = useState(0)
  const [autoCycle, setAutoCycle] = useState(false)
  const autoCycleRef = useRef(false)
  const cycleTimerRef = useRef<number>(0)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    if (isSSR || !canvasRef.current) return

    let engine: ShaderArtEngine | null = null
    let cancelled = false

    import('./engine').then(({ ShaderArtEngine }) => {
      if (cancelled || !canvasRef.current) return
      engine = new ShaderArtEngine(canvasRef.current)
      engineRef.current = engine
      setShaderCount(engine.shaderCount)
      setCurrentName(engine.shaderNames[0])
      setCurrentIndex(0)
    })

    return () => {
      cancelled = true
      if (engine) {
        engine.dispose()
        engineRef.current = null
      }
    }
  }, [isSSR])

  const goToShader = useCallback((index: number) => {
    const engine = engineRef.current
    if (!engine) return
    const count = engine.shaderCount
    const wrapped = ((index % count) + count) % count
    engine.setShader(wrapped)
    setCurrentIndex(wrapped)
    setCurrentName(engine.shaderNames[wrapped])
  }, [])

  const handlePrev = useCallback(() => {
    goToShader(currentIndex - 1)
  }, [currentIndex, goToShader])

  const handleNext = useCallback(() => {
    goToShader(currentIndex + 1)
  }, [currentIndex, goToShader])

  const handleCanvasClick = useCallback(() => {
    handleNext()
  }, [handleNext])

  // Keyboard navigation
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        goToShader((engineRef.current?.currentIndex ?? 0) - 1)
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        goToShader((engineRef.current?.currentIndex ?? 0) + 1)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [goToShader])

  // Auto-cycle
  useEffect(() => {
    autoCycleRef.current = autoCycle
    if (autoCycle) {
      const tick = () => {
        if (!autoCycleRef.current) return
        const engine = engineRef.current
        if (engine) {
          const next = (engine.currentIndex + 1) % engine.shaderCount
          engine.setShader(next)
          setCurrentIndex(next)
          setCurrentName(engine.shaderNames[next])
        }
        cycleTimerRef.current = window.setTimeout(tick, 5000)
      }
      cycleTimerRef.current = window.setTimeout(tick, 5000)
    }
    return () => {
      if (cycleTimerRef.current) {
        clearTimeout(cycleTimerRef.current)
        cycleTimerRef.current = 0
      }
    }
  }, [autoCycle])

  if (isSSR) {
    return <div className={styles.page} />
  }

  return (
    <div className={styles.page}>
      <div
        ref={canvasRef}
        className={styles.canvas}
        onClick={handleCanvasClick}
      />

      {/* Toggle panel button */}
      <button
        className={styles.panelToggle}
        onClick={() => setIsPanelOpen(prev => !prev)}
        aria-label={isPanelOpen ? 'Hide controls' : 'Show controls'}
      >
        {isPanelOpen ? '\u2715' : '\u2699'}
      </button>

      {/* Bottom bar control panel */}
      <div className={clsx(styles.panel, { [styles.panelClosed]: !isPanelOpen })}>
        <button className={styles.arrowButton} onClick={handlePrev} aria-label="Previous shader">
          &#8592;
        </button>

        <div className={styles.shaderInfo}>
          <span className={styles.shaderName}>{currentName}</span>
          <span className={styles.shaderIndex}>{currentIndex + 1} / {shaderCount}</span>
        </div>

        <button className={styles.arrowButton} onClick={handleNext} aria-label="Next shader">
          &#8594;
        </button>

        <div className={styles.divider} />

        <button
          className={clsx(styles.cycleButton, { [styles.cycleOn]: autoCycle })}
          onClick={() => setAutoCycle(prev => !prev)}
          aria-label="Toggle auto cycle"
        >
          {autoCycle ? 'AUTO' : 'AUTO'}
        </button>
      </div>

      <div className={styles.hint}>
        Click canvas or use arrow keys
      </div>
    </div>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => (
  <Seo
    title="Shader Art Gallery"
    description="Interactive gallery of GPU fragment shaders -- plasma, kaleidoscope, metaballs, fractals, voronoi, aurora, liquid metal, and neon grid"
    heroImage=""
    pathname={pathname}
  />
)

export default ShaderArtPage
