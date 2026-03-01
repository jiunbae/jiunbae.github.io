import { useRef, useEffect, useState, useCallback } from 'react'
import clsx from 'clsx'

import * as styles from './CyberFlowers.module.scss'
import type { FlowerStyle, CyberFlowerScene } from './scene'

const STYLES: { key: FlowerStyle; label: string; className: string }[] = [
  { key: 'neon', label: 'Neon', className: styles.neon },
  { key: 'holographic', label: 'Holo', className: styles.holographic },
  { key: 'retro', label: 'Retro', className: styles.retro }
]

const CyberFlowersPage = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<CyberFlowerScene | null>(null)
  const [activeStyle, setActiveStyle] = useState<FlowerStyle>('neon')
  const [accentColor, setAccentColor] = useState('#00ffff')
  const [flowerCount, setFlowerCount] = useState(0)
  const [isSSR, setIsSSR] = useState(true)
  const [autoRotate, setAutoRotate] = useState(true)
  const [rotateSpeed, setRotateSpeed] = useState(0.5)
  const [bloomStrength, setBloomStrength] = useState(1.0)
  const [showParticles, setShowParticles] = useState(true)
  const [isPanelOpen, setIsPanelOpen] = useState(true)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    if (isSSR || !canvasRef.current) return

    let scene: CyberFlowerScene | null = null
    let cancelled = false

    import('./scene').then(({ CyberFlowerScene }) => {
      if (cancelled || !canvasRef.current) return
      scene = new CyberFlowerScene(canvasRef.current)
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
    if (sceneRef.current) sceneRef.current.flowerStyle = activeStyle
  }, [activeStyle])

  useEffect(() => {
    if (sceneRef.current) sceneRef.current.accentColor = accentColor
  }, [accentColor])

  useEffect(() => {
    if (sceneRef.current) sceneRef.current.autoRotate = autoRotate
  }, [autoRotate])

  useEffect(() => {
    if (sceneRef.current) sceneRef.current.autoRotateSpeed = rotateSpeed
  }, [rotateSpeed])

  useEffect(() => {
    if (sceneRef.current) sceneRef.current.bloomStrength = bloomStrength
  }, [bloomStrength])

  useEffect(() => {
    if (sceneRef.current) sceneRef.current.particlesVisible = showParticles
  }, [showParticles])

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!sceneRef.current || !canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1
    const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1
    sceneRef.current.plantFlower(ndcX, ndcY)
    setFlowerCount(prev => prev + 1)
  }, [])

  const handleClear = useCallback(() => {
    if (sceneRef.current) {
      sceneRef.current.clearFlowers()
      setFlowerCount(0)
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
        role="img"
        aria-label="3D cyber flower garden"
      />

      <div className={clsx(styles.hint, { [styles.hidden]: flowerCount > 0 })}>
        Click anywhere to plant a flower
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
        {/* Flower style */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Style</span>
          <div className={styles.styleButtons}>
            {STYLES.map(s => (
              <button
                key={s.key}
                className={clsx(styles.styleButton, s.className, { [styles.active]: activeStyle === s.key })}
                onClick={() => setActiveStyle(s.key)}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        {/* Color picker */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Color</span>
          <input
            type="color"
            value={accentColor}
            onChange={e => setAccentColor(e.target.value)}
            className={styles.colorPicker}
            title="Accent color"
          />
        </div>

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

        {/* Rotate speed */}
        {autoRotate && (
          <div className={styles.controlGroup}>
            <span className={styles.controlLabel}>Speed</span>
            <input
              type="range"
              min="0.1"
              max="3"
              step="0.1"
              value={rotateSpeed}
              onChange={e => setRotateSpeed(parseFloat(e.target.value))}
              className={styles.slider}
            />
          </div>
        )}

        {/* Bloom */}
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

        {/* Particles */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Particles</span>
          <button
            className={clsx(styles.toggleButton, { [styles.toggleOn]: showParticles })}
            onClick={() => setShowParticles(prev => !prev)}
          >
            {showParticles ? 'ON' : 'OFF'}
          </button>
        </div>

        <div className={styles.panelDivider} />

        {/* Actions */}
        <div className={styles.controlGroup}>
          <button className={styles.actionButton} onClick={handleResetCamera}>
            Reset View
          </button>
          <button className={clsx(styles.actionButton, styles.clearButton)} onClick={handleClear}>
            Clear ({flowerCount})
          </button>
        </div>
      </div>
    </div>
  )
}


export default CyberFlowersPage
