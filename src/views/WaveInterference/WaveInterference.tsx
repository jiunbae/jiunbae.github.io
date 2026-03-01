import { useCallback, useEffect, useRef, useState } from 'react'
import type { HeadProps } from 'gatsby'
import clsx from 'clsx'
import { Seo } from '@/components'

import * as styles from './WaveInterference.module.scss'
import type { WaveInterferenceEngine, WavePalette } from './engine'

const PALETTES: { key: WavePalette; label: string }[] = [
  { key: 'neon', label: 'Neon' },
  { key: 'sunset', label: 'Sunset' },
  { key: 'mono', label: 'Mono' }
]

const WaveInterferencePage = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const engineRef = useRef<WaveInterferenceEngine | null>(null)
  const hintTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const [isSSR, setIsSSR] = useState(true)
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [hintVisible, setHintVisible] = useState(true)

  const [running, setRunning] = useState(true)
  const [palette, setPalette] = useState<WavePalette>('neon')
  const [frequency, setFrequency] = useState(1.2)
  const [wavelength, setWavelength] = useState(34)
  const [damping, setDamping] = useState(0.008)
  const [amplitude, setAmplitude] = useState(1.0)
  const [resolution, setResolution] = useState(4)
  const [sourceCount, setSourceCount] = useState(3)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    if (isSSR || !canvasRef.current) return

    let engine: WaveInterferenceEngine | null = null
    let cancelled = false

    import('./engine').then(({ WaveInterferenceEngine }) => {
      if (cancelled || !canvasRef.current) return

      engine = new WaveInterferenceEngine(canvasRef.current)
      engineRef.current = engine
      engine.setSourceCountListener(next => {
        setSourceCount(prev => (prev === next ? prev : next))
      })
      engine.setPalette(palette)
      engine.setFrequency(frequency)
      engine.setWavelength(wavelength)
      engine.setDamping(damping)
      engine.setAmplitude(amplitude)
      engine.setResolution(resolution)
      engine.setSourceCount(sourceCount)
      engine.setRunning(running)

      hintTimerRef.current = setTimeout(() => setHintVisible(false), 4200)
    })

    return () => {
      cancelled = true
      if (hintTimerRef.current) clearTimeout(hintTimerRef.current)
      if (engine) {
        engine.setSourceCountListener(null)
        engine.dispose()
        engineRef.current = null
      }
    }
  }, [isSSR])

  useEffect(() => {
    if (engineRef.current) engineRef.current.setRunning(running)
  }, [running])

  useEffect(() => {
    if (engineRef.current) engineRef.current.setPalette(palette)
  }, [palette])

  useEffect(() => {
    if (engineRef.current) engineRef.current.setFrequency(frequency)
  }, [frequency])

  useEffect(() => {
    if (engineRef.current) engineRef.current.setWavelength(wavelength)
  }, [wavelength])

  useEffect(() => {
    if (engineRef.current) engineRef.current.setDamping(damping)
  }, [damping])

  useEffect(() => {
    if (engineRef.current) engineRef.current.setAmplitude(amplitude)
  }, [amplitude])

  useEffect(() => {
    if (engineRef.current) engineRef.current.setResolution(resolution)
  }, [resolution])

  useEffect(() => {
    if (engineRef.current) engineRef.current.setSourceCount(sourceCount)
  }, [sourceCount])

  const handleReset = useCallback(() => {
    if (!engineRef.current) return
    engineRef.current.resetSources()
  }, [])

  if (isSSR) {
    return <div className={styles.page} />
  }

  return (
    <div className={styles.page}>
      <div ref={canvasRef} className={styles.canvas} />

      <div className={clsx(styles.hint, { [styles.hidden]: !hintVisible })}>
        Click to add source, drag to move, right click to remove
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
          <button className={styles.actionButton} onClick={handleReset}>
            Reset Sources
          </button>
          <button
            className={styles.secondaryButton}
            onClick={() => setSourceCount(prev => Math.max(1, prev - 1))}
          >
            - Source
          </button>
          <button
            className={styles.secondaryButton}
            onClick={() => setSourceCount(prev => Math.min(8, prev + 1))}
          >
            + Source
          </button>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Palette</span>
          <div className={styles.paletteButtons}>
            {PALETTES.map(item => (
              <button
                key={item.key}
                className={clsx(styles.paletteButton, { [styles.active]: palette === item.key })}
                onClick={() => setPalette(item.key)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Frequency</span>
          <input
            type="range"
            className={styles.slider}
            min="0.2"
            max="4"
            step="0.05"
            value={frequency}
            onChange={e => setFrequency(parseFloat(e.target.value))}
          />
          <span className={styles.value}>{frequency.toFixed(2)}</span>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Wavelength</span>
          <input
            type="range"
            className={styles.slider}
            min="8"
            max="140"
            step="1"
            value={wavelength}
            onChange={e => setWavelength(parseInt(e.target.value, 10))}
          />
          <span className={styles.value}>{wavelength}px</span>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Damping</span>
          <input
            type="range"
            className={styles.slider}
            min="0"
            max="0.04"
            step="0.001"
            value={damping}
            onChange={e => setDamping(parseFloat(e.target.value))}
          />
          <span className={styles.value}>{damping.toFixed(3)}</span>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Amplitude</span>
          <input
            type="range"
            className={styles.slider}
            min="0.2"
            max="3"
            step="0.05"
            value={amplitude}
            onChange={e => setAmplitude(parseFloat(e.target.value))}
          />
          <span className={styles.value}>{amplitude.toFixed(2)}</span>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Resolution</span>
          <input
            type="range"
            className={styles.slider}
            min="2"
            max="10"
            step="1"
            value={resolution}
            onChange={e => setResolution(parseInt(e.target.value, 10))}
          />
          <span className={styles.value}>{resolution}px</span>
        </div>

        <div className={styles.controlRow}>
          <span className={styles.controlLabel}>Sources</span>
          <div className={styles.sourceBadge}>{sourceCount}</div>
        </div>
      </div>
    </div>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => (
  <Seo
    title="Wave Interference"
    description="Interactive wave interference field where you can place and drag wave sources"
    heroImage=""
    pathname={pathname}
  />
)

export default WaveInterferencePage
