import { useRef, useEffect, useState, useCallback } from 'react'
import clsx from 'clsx'

import * as styles from './Boids.module.scss'

type BoidsEngineType = import('./engine').BoidsEngine
type MouseMode = 'attract' | 'repel' | 'none'

const COUNT_OPTIONS = [500, 1000, 2000, 3000]

const BoidsPage = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const engineRef = useRef<BoidsEngineType | null>(null)
  const [isSSR, setIsSSR] = useState(true)
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [separation, setSeparation] = useState(1.5)
  const [alignment, setAlignment] = useState(1.0)
  const [cohesion, setCohesion] = useState(1.0)
  const [perceptionRadius, setPerceptionRadius] = useState(50)
  const [boidCount, setBoidCount] = useState(1500)
  const [mouseMode, setMouseMode] = useState<MouseMode>('none')
  const [hasPredator, setHasPredator] = useState(false)
  const [hasTrails, setHasTrails] = useState(false)
  const [hintVisible, setHintVisible] = useState(true)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    if (isSSR || !canvasRef.current) return

    let engine: BoidsEngineType | null = null
    let cancelled = false

    import('./engine').then(({ BoidsEngine }) => {
      if (cancelled || !canvasRef.current) return
      engine = new BoidsEngine(canvasRef.current)
      engineRef.current = engine
      // Hide hint after a delay
      setTimeout(() => setHintVisible(false), 3000)
    })

    return () => {
      cancelled = true
      if (engine) {
        engine.dispose()
        engineRef.current = null
      }
    }
  }, [isSSR])

  // Sync parameters
  useEffect(() => {
    if (engineRef.current) engineRef.current.setSeparation(separation)
  }, [separation])

  useEffect(() => {
    if (engineRef.current) engineRef.current.setAlignment(alignment)
  }, [alignment])

  useEffect(() => {
    if (engineRef.current) engineRef.current.setCohesion(cohesion)
  }, [cohesion])

  useEffect(() => {
    if (engineRef.current) engineRef.current.setPerceptionRadius(perceptionRadius)
  }, [perceptionRadius])

  useEffect(() => {
    if (engineRef.current) engineRef.current.setMouseMode(mouseMode)
  }, [mouseMode])

  const handleCountChange = useCallback((n: number) => {
    setBoidCount(n)
    if (engineRef.current) engineRef.current.setBoidCount(n)
  }, [])

  const handleTogglePredator = useCallback(() => {
    if (engineRef.current) {
      const state = engineRef.current.togglePredator()
      setHasPredator(state)
    }
  }, [])

  const handleToggleTrails = useCallback(() => {
    if (engineRef.current) {
      const state = engineRef.current.toggleTrails()
      setHasTrails(state)
    }
  }, [])

  if (isSSR) {
    return <div className={styles.page} />
  }

  return (
    <div className={styles.page}>
      <div ref={canvasRef} className={styles.canvas} role="img" aria-label="Boids flocking simulation" />

      <div className={clsx(styles.hint, { [styles.hidden]: !hintVisible })}>
        Watch emergent flocking behavior
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
          <span className={styles.controlLabel}>Separation</span>
          <input
            type="range"
            min="0"
            max="5"
            step="0.1"
            value={separation}
            onChange={e => setSeparation(parseFloat(e.target.value))}
            className={styles.slider}
          />
        </div>

        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Alignment</span>
          <input
            type="range"
            min="0"
            max="5"
            step="0.1"
            value={alignment}
            onChange={e => setAlignment(parseFloat(e.target.value))}
            className={styles.slider}
          />
        </div>

        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Cohesion</span>
          <input
            type="range"
            min="0"
            max="5"
            step="0.1"
            value={cohesion}
            onChange={e => setCohesion(parseFloat(e.target.value))}
            className={styles.slider}
          />
        </div>

        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Perception</span>
          <input
            type="range"
            min="20"
            max="200"
            step="5"
            value={perceptionRadius}
            onChange={e => setPerceptionRadius(parseFloat(e.target.value))}
            className={styles.slider}
          />
        </div>

        <div className={styles.panelDivider} />

        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Boids</span>
          <div className={styles.countButtons}>
            {COUNT_OPTIONS.map(n => (
              <button
                key={n}
                className={clsx(styles.countButton, { [styles.active]: boidCount === n })}
                onClick={() => handleCountChange(n)}
              >
                {n >= 1000 ? `${n / 1000}k` : n}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Mouse</span>
          <div className={styles.modeButtons}>
            {(['attract', 'repel', 'none'] as MouseMode[]).map(mode => (
              <button
                key={mode}
                className={clsx(styles.modeButton, { [styles.active]: mouseMode === mode })}
                onClick={() => setMouseMode(mode)}
              >
                {mode[0].toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.panelDivider} />

        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Predator</span>
          <button
            className={clsx(styles.toggleButton, { [styles.toggleOn]: hasPredator })}
            onClick={handleTogglePredator}
          >
            {hasPredator ? 'ON' : 'OFF'}
          </button>
        </div>

        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Trails</span>
          <button
            className={clsx(styles.toggleButton, { [styles.toggleOn]: hasTrails })}
            onClick={handleToggleTrails}
          >
            {hasTrails ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>
    </div>
  )
}


export default BoidsPage
