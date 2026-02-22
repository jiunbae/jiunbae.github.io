import { useRef, useEffect, useState, useCallback } from 'react'
import type { HeadProps } from 'gatsby'
import clsx from 'clsx'
import { Seo } from '@/components'

import * as styles from './TerrainGen.module.scss'
import type { TerrainEngine } from './engine'

const TerrainGenPage = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const engineRef = useRef<TerrainEngine | null>(null)
  const [isSSR, setIsSSR] = useState(true)
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [autoRotate, setAutoRotate] = useState(true)
  const [noiseScale, setNoiseScale] = useState(2.0)
  const [octaves, setOctaves] = useState(5)
  const [erosion, setErosion] = useState(500)
  const [waterLevel, setWaterLevel] = useState(0.35)
  const [currentSeed, setCurrentSeed] = useState(0)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    if (isSSR || !canvasRef.current) return

    let engine: TerrainEngine | null = null
    let cancelled = false

    import('./engine').then(({ TerrainEngine }) => {
      if (cancelled || !canvasRef.current) return
      engine = new TerrainEngine(canvasRef.current)
      engineRef.current = engine
      setCurrentSeed(engine.seed)
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
    if (engineRef.current) engineRef.current.autoRotate = autoRotate
  }, [autoRotate])

  const handleGenerate = useCallback(() => {
    if (!engineRef.current) return
    engineRef.current.generate()
    setCurrentSeed(engineRef.current.seed)
  }, [])

  const handleNoiseScale = useCallback((v: number) => {
    setNoiseScale(v)
    if (engineRef.current) {
      engineRef.current.setNoiseScale(v)
      setCurrentSeed(engineRef.current.seed)
    }
  }, [])

  const handleOctaves = useCallback((v: number) => {
    setOctaves(v)
    if (engineRef.current) {
      engineRef.current.setOctaves(v)
      setCurrentSeed(engineRef.current.seed)
    }
  }, [])

  const handleErosion = useCallback((v: number) => {
    setErosion(v)
    if (engineRef.current) {
      engineRef.current.setErosionSteps(v)
      setCurrentSeed(engineRef.current.seed)
    }
  }, [])

  const handleWaterLevel = useCallback((v: number) => {
    setWaterLevel(v)
    if (engineRef.current) {
      engineRef.current.setWaterLevel(v)
    }
  }, [])

  if (isSSR) {
    return <div className={styles.page} />
  }

  return (
    <div className={styles.page}>
      <div ref={canvasRef} className={styles.canvas} />

      <button
        className={styles.panelToggle}
        onClick={() => setIsPanelOpen(prev => !prev)}
        aria-label={isPanelOpen ? 'Hide controls' : 'Show controls'}
      >
        {isPanelOpen ? '\u2715' : '\u2699'}
      </button>

      <div className={clsx(styles.panel, { [styles.panelClosed]: !isPanelOpen })}>
        {/* Noise Scale */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Scale</span>
          <input
            type="range"
            min="0.5"
            max="5"
            step="0.1"
            value={noiseScale}
            onChange={e => handleNoiseScale(parseFloat(e.target.value))}
            className={styles.slider}
          />
          <span className={styles.stat}>{noiseScale.toFixed(1)}</span>
        </div>

        {/* Octaves */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Detail</span>
          <input
            type="range"
            min="1"
            max="8"
            step="1"
            value={octaves}
            onChange={e => handleOctaves(parseInt(e.target.value))}
            className={styles.slider}
          />
          <span className={styles.stat}>{octaves}</span>
        </div>

        {/* Erosion */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Erosion</span>
          <input
            type="range"
            min="0"
            max="5000"
            step="100"
            value={erosion}
            onChange={e => handleErosion(parseInt(e.target.value))}
            className={styles.slider}
          />
          <span className={styles.stat}>{erosion}</span>
        </div>

        {/* Water Level */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Water</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={waterLevel}
            onChange={e => handleWaterLevel(parseFloat(e.target.value))}
            className={styles.slider}
          />
          <span className={styles.stat}>{(waterLevel * 100).toFixed(0)}%</span>
        </div>

        <div className={styles.panelDivider} />

        {/* Auto rotate */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Rotate</span>
          <button
            className={clsx(styles.toggleButton, { [styles.toggleOn]: autoRotate })}
            onClick={() => setAutoRotate(prev => !prev)}
          >
            {autoRotate ? 'ON' : 'OFF'}
          </button>
        </div>

        <div className={styles.panelDivider} />

        {/* Generate */}
        <div className={styles.controlGroup}>
          <button className={styles.actionButton} onClick={handleGenerate}>
            Generate New
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
    title="Terrain Generator"
    description="Procedural 3D terrain with simplex noise, hydraulic erosion simulation, and biome coloring"
    heroImage=""
    pathname={pathname}
  />
)

export default TerrainGenPage
