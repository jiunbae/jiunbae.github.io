import { useRef, useEffect, useState, useCallback } from 'react'
import type { HeadProps } from 'gatsby'
import clsx from 'clsx'
import { Seo } from '@/components'

import * as styles from './GenerativeArt.module.scss'
import type { ArtMode, PaletteInfo, GenerativeArtEngine } from './engine'

const MODES: { key: ArtMode; label: string }[] = [
  { key: 'flowField', label: 'Flow Field' },
  { key: 'fractalTree', label: 'Fractal Tree' },
  { key: 'circlePacking', label: 'Circle Pack' },
  { key: 'voronoi', label: 'Voronoi' }
]

const GenerativeArtPage = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const engineRef = useRef<GenerativeArtEngine | null>(null)
  const palettesRef = useRef<PaletteInfo[]>([])
  const [isSSR, setIsSSR] = useState(true)
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [activeMode, setActiveMode] = useState<ArtMode>('flowField')
  const [activePaletteIdx, setActivePaletteIdx] = useState(0)
  const [currentSeed, setCurrentSeed] = useState(0)
  const [hasGenerated, setHasGenerated] = useState(false)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    if (isSSR || !canvasRef.current) return

    let engine: GenerativeArtEngine | null = null
    let cancelled = false

    import('./engine').then(({ GenerativeArtEngine, PALETTES }) => {
      if (cancelled || !canvasRef.current) return
      palettesRef.current = PALETTES
      engine = new GenerativeArtEngine(canvasRef.current)
      engineRef.current = engine
      setCurrentSeed(engine.seed)
      setHasGenerated(true)
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
      engineRef.current.setMode(activeMode)
      setCurrentSeed(engineRef.current.seed)
    }
  }, [activeMode])

  useEffect(() => {
    if (engineRef.current && palettesRef.current[activePaletteIdx]) {
      engineRef.current.setPalette(palettesRef.current[activePaletteIdx])
      setCurrentSeed(engineRef.current.seed)
    }
  }, [activePaletteIdx])

  const handleGenerate = useCallback(() => {
    if (!engineRef.current) return
    engineRef.current.generate()
    setCurrentSeed(engineRef.current.seed)
    setHasGenerated(true)
  }, [])

  const handleCanvasClick = useCallback(() => {
    handleGenerate()
  }, [handleGenerate])

  const handleExport = useCallback(() => {
    if (engineRef.current) engineRef.current.exportPNG()
  }, [])

  if (isSSR) {
    return <div className={styles.page} />
  }

  return (
    <div className={styles.page}>
      <div
        ref={canvasRef}
        className={styles.canvas}
        onClick={handleCanvasClick}
        role="img"
        aria-label="Generative art canvas"
      />

      <div className={clsx(styles.hint, { [styles.hidden]: hasGenerated })}>
        Click anywhere to generate art
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
          <span className={styles.controlLabel}>Mode</span>
          <div className={styles.modeButtons}>
            {MODES.map(m => (
              <button
                key={m.key}
                className={clsx(styles.modeButton, { [styles.active]: activeMode === m.key })}
                onClick={() => setActiveMode(m.key)}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Palette</span>
          <div className={styles.palettes}>
            {(palettesRef.current.length > 0 ? palettesRef.current : [
              { name: 'neonNoir', colors: ['#ff006e', '#fb5607', '#ffbe0b', '#8338ec', '#3a86ff'] },
              { name: 'ocean', colors: ['#0077b6', '#00b4d8', '#90e0ef', '#caf0f8', '#03045e'] },
              { name: 'sunset', colors: ['#ff595e', '#ff924c', '#ffca3a', '#c77dff', '#9b5de5'] },
              { name: 'forest', colors: ['#606c38', '#283618', '#dda15e', '#bc6c25', '#fefae0'] },
              { name: 'mono', colors: ['#f8f9fa', '#adb5bd', '#6c757d', '#343a40', '#212529'] }
            ] as { name: string; colors: string[] }[]).map((p, i) => (
              <button
                key={p.name}
                className={clsx(styles.paletteSwatch, { [styles.activePalette]: activePaletteIdx === i })}
                onClick={() => setActivePaletteIdx(i)}
                title={p.name}
                style={{
                  background: `linear-gradient(135deg, ${p.colors[0]} 0%, ${p.colors[2]} 50%, ${p.colors[4]} 100%)`
                }}
              />
            ))}
          </div>
        </div>

        <div className={styles.panelDivider} />

        <div className={styles.controlGroup}>
          <button className={styles.actionButton} onClick={handleGenerate}>
            Generate
          </button>
          <button className={clsx(styles.actionButton, styles.exportButton)} onClick={handleExport}>
            Export PNG
          </button>
        </div>

        <div className={styles.seedDisplay}>
          seed: {currentSeed}
        </div>
      </div>
    </div>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => (
  <Seo
    title="Generative Art Studio"
    description="Algorithmic art generator — flow fields, fractals, circle packing, Voronoi. Every click creates unique art"
    heroImage=""
    pathname={pathname}
  />
)

export default GenerativeArtPage
