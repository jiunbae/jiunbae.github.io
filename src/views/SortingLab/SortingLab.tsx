import { useCallback, useEffect, useRef, useState } from 'react'
import type { HeadProps } from 'gatsby'
import clsx from 'clsx'
import { Seo } from '@/components'

import * as styles from './SortingLab.module.scss'
import type {
  Distribution,
  SortAlgorithm,
  SortSnapshot,
  SortingLabEngine
} from './engine'

const ALGORITHMS: { key: SortAlgorithm; label: string }[] = [
  { key: 'quick', label: 'Quick' },
  { key: 'bubble', label: 'Bubble' },
  { key: 'selection', label: 'Selection' },
  { key: 'insertion', label: 'Insertion' }
]

const DISTRIBUTIONS: { key: Distribution; label: string }[] = [
  { key: 'random', label: 'Random' },
  { key: 'reversed', label: 'Reversed' },
  { key: 'nearlySorted', label: 'Nearly Sorted' }
]

const EMPTY_SNAPSHOT: SortSnapshot = {
  running: false,
  done: false,
  algorithm: 'quick',
  size: 120,
  distribution: 'random',
  comparisons: 0,
  swaps: 0,
  operations: 0,
  totalOperations: 0,
  progress: 0
}

function isSameSnapshot(a: SortSnapshot, b: SortSnapshot): boolean {
  return (
    a.running === b.running &&
    a.done === b.done &&
    a.algorithm === b.algorithm &&
    a.size === b.size &&
    a.distribution === b.distribution &&
    a.comparisons === b.comparisons &&
    a.swaps === b.swaps &&
    a.operations === b.operations &&
    a.totalOperations === b.totalOperations &&
    a.progress === b.progress
  )
}

const SortingLabPage = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const engineRef = useRef<SortingLabEngine | null>(null)
  const hintTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const [isSSR, setIsSSR] = useState(true)
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [hintVisible, setHintVisible] = useState(true)

  const [algorithm, setAlgorithm] = useState<SortAlgorithm>('quick')
  const [distribution, setDistribution] = useState<Distribution>('random')
  const [size, setSize] = useState(120)
  const [speed, setSpeed] = useState(80)
  const [snapshot, setSnapshot] = useState<SortSnapshot>(EMPTY_SNAPSHOT)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    if (isSSR || !canvasRef.current) return

    let engine: SortingLabEngine | null = null
    let cancelled = false

    import('./engine').then(({ SortingLabEngine }) => {
      if (cancelled || !canvasRef.current) return

      engine = new SortingLabEngine(canvasRef.current)
      engineRef.current = engine
      engine.setSnapshotListener(next => {
        setSnapshot(prev => (isSameSnapshot(prev, next) ? prev : next))
      })
      engine.setDistribution(distribution)
      engine.setAlgorithm(algorithm)
      engine.setSize(size)
      engine.setSpeed(speed)

      hintTimerRef.current = setTimeout(() => setHintVisible(false), 3600)
    })

    return () => {
      cancelled = true
      if (hintTimerRef.current) clearTimeout(hintTimerRef.current)
      if (engine) {
        engine.setSnapshotListener(null)
        engine.dispose()
        engineRef.current = null
      }
    }
  }, [isSSR])

  useEffect(() => {
    if (!engineRef.current) return
    engineRef.current.setAlgorithm(algorithm)
  }, [algorithm])

  useEffect(() => {
    if (!engineRef.current) return
    engineRef.current.setDistribution(distribution)
  }, [distribution])

  useEffect(() => {
    if (!engineRef.current) return
    engineRef.current.setSize(size)
  }, [size])

  useEffect(() => {
    if (!engineRef.current) return
    engineRef.current.setSpeed(speed)
  }, [speed])

  const handleShuffle = useCallback(() => {
    if (!engineRef.current) return
    engineRef.current.shuffle(distribution)
  }, [distribution])

  const handleToggleRun = useCallback(() => {
    if (!engineRef.current) return
    if (snapshot.running) {
      engineRef.current.pause()
    } else {
      engineRef.current.start()
    }
  }, [snapshot.running])

  const handleStep = useCallback(() => {
    if (!engineRef.current) return
    engineRef.current.step(1)
  }, [])

  if (isSSR) {
    return <div className={styles.page} />
  }

  const progressPercent = Math.round(snapshot.progress * 100)

  return (
    <div className={styles.page}>
      <div ref={canvasRef} className={styles.canvas} />

      <div className={clsx(styles.hint, { [styles.hidden]: !hintVisible })}>
        Shuffle data then hit start
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
          <button className={styles.actionButton} onClick={handleToggleRun}>
            {snapshot.running ? 'Pause' : 'Start'}
          </button>
          <button className={styles.secondaryButton} onClick={handleStep}>
            Step
          </button>
          <button className={styles.secondaryButton} onClick={handleShuffle}>
            Shuffle
          </button>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Algorithm</span>
          <div className={styles.algoButtons}>
            {ALGORITHMS.map(item => (
              <button
                key={item.key}
                className={clsx(styles.algoButton, { [styles.active]: algorithm === item.key })}
                onClick={() => setAlgorithm(item.key)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Distribution</span>
          <div className={styles.algoButtons}>
            {DISTRIBUTIONS.map(item => (
              <button
                key={item.key}
                className={clsx(styles.algoButton, { [styles.active]: distribution === item.key })}
                onClick={() => setDistribution(item.key)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Size</span>
          <input
            className={styles.slider}
            type="range"
            min="24"
            max="280"
            step="1"
            value={size}
            onChange={e => setSize(parseInt(e.target.value, 10))}
          />
          <span className={styles.value}>{size}</span>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Speed</span>
          <input
            className={styles.slider}
            type="range"
            min="1"
            max="260"
            step="1"
            value={speed}
            onChange={e => setSpeed(parseInt(e.target.value, 10))}
          />
          <span className={styles.value}>{speed} op/f</span>
        </div>

        <div className={styles.stats}>
          <div className={styles.statLine}>
            <span className={styles.statLabel}>Comparisons</span>
            <span className={styles.statValue}>{snapshot.comparisons}</span>
          </div>
          <div className={styles.statLine}>
            <span className={styles.statLabel}>Swaps</span>
            <span className={styles.statValue}>{snapshot.swaps}</span>
          </div>
          <div className={styles.statLine}>
            <span className={styles.statLabel}>Progress</span>
            <span className={styles.statValue}>{progressPercent}%</span>
          </div>
          <div className={styles.progressBar}>
            <div className={styles.progressFill} style={{ width: `${progressPercent}%` }} />
          </div>
        </div>
      </div>
    </div>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => (
  <Seo
    title="Sorting Lab"
    description="Sorting algorithm visualizer with speed, input distribution, and algorithm controls"
    heroImage=""
    pathname={pathname}
  />
)

export default SortingLabPage
