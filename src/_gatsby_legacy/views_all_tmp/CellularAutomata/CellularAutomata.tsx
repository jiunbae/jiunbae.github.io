import { useCallback, useEffect, useRef, useState } from 'react'
import type { HeadProps } from 'gatsby'
import clsx from 'clsx'
import { Seo } from '@/components'

import * as styles from './CellularAutomata.module.scss'
import type { CellularAutomataEngine, RulePreset } from './engine'

const RULES: { key: RulePreset; label: string }[] = [
  { key: 'life', label: 'Life' },
  { key: 'highlife', label: 'HighLife' },
  { key: 'seeds', label: 'Seeds' }
]

const CellularAutomataPage = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const engineRef = useRef<CellularAutomataEngine | null>(null)
  const hintTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const [isSSR, setIsSSR] = useState(true)
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [hintVisible, setHintVisible] = useState(true)

  const [running, setRunning] = useState(true)
  const [rule, setRule] = useState<RulePreset>('life')
  const [wrapEdges, setWrapEdges] = useState(true)
  const [cellSize, setCellSize] = useState(8)
  const [speed, setSpeed] = useState(14)
  const [density, setDensity] = useState(0.24)
  const [generation, setGeneration] = useState(0)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    if (isSSR || !canvasRef.current) return

    let engine: CellularAutomataEngine | null = null
    let cancelled = false

    import('./engine').then(({ CellularAutomataEngine }) => {
      if (cancelled || !canvasRef.current) return

      engine = new CellularAutomataEngine(canvasRef.current)
      engineRef.current = engine

      engine.setGenerationListener(next => {
        setGeneration(prev => (prev === next ? prev : next))
      })
      engine.setRule(rule)
      engine.setWrapEdges(wrapEdges)
      engine.setCellSize(cellSize)
      engine.setSpeed(speed)
      engine.setRunning(running)
      engine.randomize(density)

      hintTimerRef.current = setTimeout(() => setHintVisible(false), 3800)
    })

    return () => {
      cancelled = true
      if (hintTimerRef.current) clearTimeout(hintTimerRef.current)
      if (engine) {
        engine.setGenerationListener(null)
        engine.dispose()
        engineRef.current = null
      }
    }
  }, [isSSR])

  useEffect(() => {
    if (engineRef.current) {
      engineRef.current.setRule(rule)
    }
  }, [rule])

  useEffect(() => {
    if (engineRef.current) {
      engineRef.current.setWrapEdges(wrapEdges)
    }
  }, [wrapEdges])

  useEffect(() => {
    if (engineRef.current) {
      engineRef.current.setCellSize(cellSize)
    }
  }, [cellSize])

  useEffect(() => {
    if (engineRef.current) {
      engineRef.current.setSpeed(speed)
    }
  }, [speed])

  useEffect(() => {
    if (engineRef.current) {
      engineRef.current.setRunning(running)
    }
  }, [running])

  const handleRandomize = useCallback(() => {
    if (!engineRef.current) return
    engineRef.current.randomize(density)
  }, [density])

  const handleClear = useCallback(() => {
    if (!engineRef.current) return
    engineRef.current.clear()
  }, [])

  const handleStep = useCallback(() => {
    if (!engineRef.current) return
    engineRef.current.step()
  }, [])

  if (isSSR) {
    return <div className={styles.page} />
  }

  return (
    <div className={styles.page}>
      <div ref={canvasRef} className={styles.canvas} />

      <div className={clsx(styles.hint, { [styles.hidden]: !hintVisible })}>
        Draw cells to seed patterns
      </div>

      <button
        className={styles.panelToggle}
        onClick={() => setIsPanelOpen(prev => !prev)}
        aria-label={isPanelOpen ? 'Hide controls' : 'Show controls'}
      >
        {isPanelOpen ? 'X' : 'O'}
      </button>

      <div className={clsx(styles.panel, { [styles.panelClosed]: !isPanelOpen })}>
        <div className={styles.buttonRow}>
          <button className={styles.actionButton} onClick={() => setRunning(prev => !prev)}>
            {running ? 'Pause' : 'Play'}
          </button>
          <button className={styles.secondaryButton} onClick={handleStep}>
            Step
          </button>
          <button className={styles.secondaryButton} onClick={handleRandomize}>
            Random
          </button>
          <button className={styles.secondaryButton} onClick={handleClear}>
            Clear
          </button>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Rule</span>
          <div className={styles.ruleButtons}>
            {RULES.map(item => (
              <button
                key={item.key}
                className={clsx(styles.ruleButton, { [styles.active]: rule === item.key })}
                onClick={() => setRule(item.key)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Cell Size</span>
          <input
            className={styles.slider}
            type="range"
            min="5"
            max="20"
            step="1"
            value={cellSize}
            onChange={e => setCellSize(parseInt(e.target.value, 10))}
          />
          <span className={styles.stat}>{cellSize}px</span>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Speed</span>
          <input
            className={styles.slider}
            type="range"
            min="2"
            max="30"
            step="1"
            value={speed}
            onChange={e => setSpeed(parseInt(e.target.value, 10))}
          />
          <span className={styles.stat}>{speed} fps</span>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Density</span>
          <input
            className={styles.slider}
            type="range"
            min="0.05"
            max="0.7"
            step="0.01"
            value={density}
            onChange={e => setDensity(parseFloat(e.target.value))}
          />
          <span className={styles.stat}>{Math.round(density * 100)}%</span>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Wrap</span>
          <button
            className={clsx(styles.toggleButton, { [styles.toggleOn]: wrapEdges })}
            onClick={() => setWrapEdges(prev => !prev)}
          >
            {wrapEdges ? 'ON' : 'OFF'}
          </button>
          <span className={styles.stat}>Gen {generation}</span>
        </div>

        <div className={styles.instruction}>
          Left click paints cells. Click a live cell to erase. Use Random to reseed quickly.
        </div>
      </div>
    </div>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => (
  <Seo
    title="Cellular Automata Lab"
    description="Interactive cellular automata playground with Life, HighLife, and Seeds rules"
    heroImage=""
    pathname={pathname}
  />
)

export default CellularAutomataPage
