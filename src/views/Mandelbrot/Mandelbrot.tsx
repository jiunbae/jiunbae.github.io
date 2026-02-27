import { useRef, useEffect, useState, useCallback } from 'react'

import type { HeadProps } from 'gatsby'
import clsx from 'clsx'
import { Seo } from '@/components'

import * as styles from './Mandelbrot.module.scss'

type MandelbrotEngine = import('./engine').MandelbrotEngine

const PALETTE_COLORS = [
  ['#1a1a6e', '#4488ff', '#ffffff', '#ff8844', '#1a1a6e'], // Classic
  ['#000000', '#cc3300', '#ff8800', '#ffff44', '#ffffff'], // Fire
  ['#001133', '#003388', '#0088cc', '#44ddff', '#ffffff'], // Ocean
  ['#330044', '#cc00ff', '#00ffcc', '#88ff00', '#ffff00'], // Neon
  ['#000000', '#444455', '#8888aa', '#ccccdd', '#ffffff'], // Mono
]

const PALETTE_NAMES = ['Classic', 'Fire', 'Ocean', 'Neon', 'Mono']

const MandelbrotPage = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const engineRef = useRef<MandelbrotEngine | null>(null)
  const [isSSR, setIsSSR] = useState(true)
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [centerX, setCenterX] = useState(-0.5)
  const [centerY, setCenterY] = useState(0.0)
  const [zoomLevel, setZoomLevel] = useState(1.5)
  const [palette, setPalette] = useState(0)
  const [maxIter, setMaxIter] = useState(200)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    if (isSSR || !canvasRef.current) return

    let engine: MandelbrotEngine | null = null
    let cancelled = false

    import('./engine').then(({ MandelbrotEngine }) => {
      if (cancelled || !canvasRef.current) return
      engine = new MandelbrotEngine(canvasRef.current)
      // H-7 fix: callback-based view updates instead of rAF polling
      engine.onViewChange = (cx, cy, z) => {
        setCenterX(cx)
        setCenterY(cy)
        setZoomLevel(z)
      }
      engineRef.current = engine
    })

    return () => {
      cancelled = true
      if (engine) {
        engine.onViewChange = null
        engine.dispose()
        engineRef.current = null
      }
    }
  }, [isSSR])

  const handlePaletteChange = useCallback((index: number) => {
    setPalette(index)
    if (engineRef.current) engineRef.current.setPalette(index)
  }, [])

  const handleIterChange = useCallback((value: number) => {
    setMaxIter(value)
    if (engineRef.current) engineRef.current.setMaxIterations(value)
  }, [])

  const handleReset = useCallback(() => {
    if (engineRef.current) {
      engineRef.current.resetView()
      setMaxIter(200)
      engineRef.current.setMaxIterations(200)
    }
  }, [])

  const formatCoord = (v: number) => {
    if (Math.abs(v) < 0.0001 && v !== 0) return v.toExponential(6)
    return v.toFixed(8)
  }

  const formatZoom = (z: number) => {
    const magnification = 1.5 / z
    if (magnification >= 1e6) return magnification.toExponential(2) + 'x'
    if (magnification >= 1000) return (magnification / 1000).toFixed(1) + 'Kx'
    return magnification.toFixed(1) + 'x'
  }

  if (isSSR) {
    return <div className={styles.page} />
  }

  return (
    <div className={styles.page}>
      <div ref={canvasRef} className={styles.canvas} role="img" aria-label="Mandelbrot fractal explorer" />

      {/* Toggle panel button */}
      <button
        className={styles.panelToggle}
        onClick={() => setIsPanelOpen(prev => !prev)}
        aria-label={isPanelOpen ? 'Hide controls' : 'Show controls'}
      >
        {isPanelOpen ? '\u2715' : '\u2699'}
      </button>

      {/* Control panel - top right */}
      <div className={clsx(styles.panel, { [styles.panelClosed]: !isPanelOpen })}>
        {/* Coordinates */}
        <div className={styles.coordSection}>
          <div className={styles.coordRow}>
            <span className={styles.coordLabel}>Re</span>
            <span className={styles.coordValue}>{formatCoord(centerX)}</span>
          </div>
          <div className={styles.coordRow}>
            <span className={styles.coordLabel}>Im</span>
            <span className={styles.coordValue}>{formatCoord(centerY)}</span>
          </div>
          <div className={styles.coordRow}>
            <span className={styles.coordLabel}>Zoom</span>
            <span className={styles.coordValue}>{formatZoom(zoomLevel)}</span>
          </div>
        </div>

        <div className={styles.panelDivider} />

        {/* Palette selector */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Palette</span>
          <div className={styles.paletteRow}>
            {PALETTE_COLORS.map((colors, i) => (
              <button
                key={i}
                className={clsx(styles.paletteSwatch, { [styles.paletteActive]: palette === i })}
                onClick={() => handlePaletteChange(i)}
                aria-label={`${PALETTE_NAMES[i]} palette`}
                title={PALETTE_NAMES[i]}
              >
                <div
                  className={styles.swatchGradient}
                  style={{
                    background: `linear-gradient(90deg, ${colors.join(', ')})`,
                  }}
                />
              </button>
            ))}
          </div>
        </div>

        {/* Iterations */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Iterations</span>
          <div className={styles.sliderRow}>
            <input
              type="range"
              min="50"
              max="1000"
              step="10"
              value={maxIter}
              onChange={e => handleIterChange(parseInt(e.target.value, 10))}
              className={styles.slider}
            />
            <span className={styles.sliderValue}>{maxIter}</span>
          </div>
        </div>

        <div className={styles.panelDivider} />

        {/* Reset */}
        <div className={styles.controlGroup}>
          <button className={styles.actionButton} onClick={handleReset}>
            Reset View
          </button>
        </div>
      </div>

      <div className={styles.hint}>
        Scroll to zoom, drag to pan
      </div>
    </div>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => (
  <Seo
    title="Mandelbrot Explorer"
    description="Interactive Mandelbrot set explorer with smooth coloring, multiple palettes, and deep zoom via double-emulated precision"
    heroImage=""
    pathname={pathname}
  />
)

export default MandelbrotPage
