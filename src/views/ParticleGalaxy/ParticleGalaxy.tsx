import { useRef, useEffect, useState, useCallback } from 'react'
import type { HeadProps } from 'gatsby'
import clsx from 'clsx'
import { Seo } from '@/components'

import * as styles from './ParticleGalaxy.module.scss'
import type { ParticleGalaxyScene } from './scene'

const ParticleGalaxyPage = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<ParticleGalaxyScene | null>(null)
  const [isSSR, setIsSSR] = useState(true)
  const [autoRotate, setAutoRotate] = useState(true)
  const [bloomStrength, setBloomStrength] = useState(1.5)
  const [attractorCount, setAttractorCount] = useState(0)
  const [isPanelOpen, setIsPanelOpen] = useState(true)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    if (isSSR || !canvasRef.current) return

    let scene: ParticleGalaxyScene | null = null
    let cancelled = false

    import('./scene').then(({ ParticleGalaxyScene }) => {
      if (cancelled || !canvasRef.current) return
      scene = new ParticleGalaxyScene(canvasRef.current)
      sceneRef.current = scene
    })

    return () => {
      cancelled = true
      if (scene) {
        scene.dispose()
        sceneRef.current = null
      }
    }
  }, [isSSR])

  useEffect(() => {
    if (sceneRef.current) sceneRef.current.autoRotate = autoRotate
  }, [autoRotate])

  useEffect(() => {
    if (sceneRef.current) sceneRef.current.bloomStrength = bloomStrength
  }, [bloomStrength])

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!sceneRef.current || !canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1
    const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1
    sceneRef.current.addAttractor(ndcX, ndcY)
    setAttractorCount(sceneRef.current.attractorCount)
  }, [])

  const handleClearAttractors = useCallback(() => {
    if (sceneRef.current) {
      sceneRef.current.clearAttractors()
      setAttractorCount(0)
    }
  }, [])

  const handleResetCamera = useCallback(() => {
    if (sceneRef.current) sceneRef.current.resetCamera()
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
      />

      <div className={clsx(styles.hint, { [styles.hidden]: attractorCount > 0 })}>
        Click to create gravitational wells
      </div>

      {/* Toggle panel button */}
      <button
        className={styles.panelToggle}
        onClick={() => setIsPanelOpen(prev => !prev)}
        aria-label={isPanelOpen ? 'Hide controls' : 'Show controls'}
      >
        {isPanelOpen ? '\u2715' : '\u2699'}
      </button>

      {/* Control panel */}
      <div className={clsx(styles.panel, { [styles.panelClosed]: !isPanelOpen })}>
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

        {/* Bloom strength */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Bloom</span>
          <input
            type="range"
            min="0"
            max="3"
            step="0.1"
            value={bloomStrength}
            onChange={e => setBloomStrength(parseFloat(e.target.value))}
            className={styles.slider}
          />
        </div>

        {/* Attractor count display */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Attractors</span>
          <span className={styles.stat}>{attractorCount} / 12</span>
        </div>

        {/* Particle count display */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Particles</span>
          <span className={styles.stat}>80,000</span>
        </div>

        <div className={styles.panelDivider} />

        {/* Actions */}
        <div className={styles.controlGroup}>
          <button className={styles.actionButton} onClick={handleResetCamera}>
            Reset View
          </button>
          <button
            className={clsx(styles.actionButton, styles.clearButton)}
            onClick={handleClearAttractors}
          >
            Clear Wells
          </button>
        </div>
      </div>
    </div>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => (
  <Seo
    title="Particle Galaxy"
    description="Interactive galaxy simulation with 80K GPU particles, spiral arms, and gravitational attractors"
    heroImage=""
    pathname={pathname}
  />
)

export default ParticleGalaxyPage
