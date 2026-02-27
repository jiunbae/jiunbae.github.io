import { useRef, useEffect, useState, useCallback } from 'react'
import type { HeadProps } from 'gatsby'
import clsx from 'clsx'
import { Seo } from '@/components'

import * as styles from './AudioVisualizer.module.scss'
import type { VisualizationMode, AudioVisualizerScene } from './scene'

const MODES: { key: VisualizationMode; label: string }[] = [
  { key: 'terrain', label: 'Terrain' },
  { key: 'radial', label: 'Radial' },
  { key: 'waveform', label: 'Waveform' }
]

const AudioVisualizerPage = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<AudioVisualizerScene | null>(null)
  const [isSSR, setIsSSR] = useState(true)
  const [activeMode, setActiveMode] = useState<VisualizationMode>('terrain')
  const [bloomStrength, setBloomStrength] = useState(1.5)
  const [autoRotate, setAutoRotate] = useState(true)
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [isDragging, setIsDragging] = useState(false)
  const [sourceName, setSourceName] = useState<string | null>(null)
  const [micError, setMicError] = useState<string | null>(null)
  const dragCounter = useRef(0)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    if (isSSR || !canvasRef.current) return

    let scene: AudioVisualizerScene | null = null
    let cancelled = false

    import('./scene').then(({ AudioVisualizerScene }) => {
      if (cancelled || !canvasRef.current) return
      scene = new AudioVisualizerScene(canvasRef.current)
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

  // Sync mode
  useEffect(() => {
    if (sceneRef.current) sceneRef.current.setMode(activeMode)
  }, [activeMode])

  // Sync bloom
  useEffect(() => {
    if (sceneRef.current) sceneRef.current.bloomStrength = bloomStrength
  }, [bloomStrength])

  // Sync auto-rotate
  useEffect(() => {
    if (sceneRef.current) sceneRef.current.autoRotate = autoRotate
  }, [autoRotate])

  // --- Audio source handlers ---

  const handleFileDrop = useCallback(async (file: File) => {
    if (!sceneRef.current) return
    try {
      await sceneRef.current.loadAudioFile(file)
      setSourceName(file.name)
    } catch (err) {
      console.error('Failed to load audio file:', err)
    }
  }, [])

  const handleMic = useCallback(async () => {
    if (!sceneRef.current) return
    setMicError(null)
    try {
      await sceneRef.current.startMic()
      setSourceName('Microphone')
    } catch (err) {
      const message = err instanceof DOMException && err.name === 'NotAllowedError'
        ? 'Microphone access denied'
        : 'Microphone error'
      setMicError(message)
      console.error('Microphone access error:', err)
    }
  }, [])

  const handleStop = useCallback(() => {
    if (sceneRef.current) sceneRef.current.stopAudio()
    setSourceName(null)
  }, [])

  // --- Drag & Drop ---

  const onDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounter.current += 1
    if (dragCounter.current === 1) setIsDragging(true)
  }, [])

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounter.current -= 1
    if (dragCounter.current <= 0) {
      dragCounter.current = 0
      setIsDragging(false)
    }
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      dragCounter.current = 0
      setIsDragging(false)

      const files = e.dataTransfer?.files
      if (files && files.length > 0) {
        const file = files[0]
        if (file.type.startsWith('audio/')) {
          handleFileDrop(file)
        }
      }
    },
    [handleFileDrop]
  )

  // Also allow click to pick a file
  const handleFileClick = useCallback(() => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'audio/*'
    input.onchange = () => {
      const file = input.files?.[0]
      if (file) handleFileDrop(file)
    }
    input.click()
  }, [handleFileDrop])

  if (isSSR) {
    return <div className={styles.page} />
  }

  return (
    <div
      className={styles.page}
      onDragEnter={onDragEnter}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <div ref={canvasRef} className={styles.canvas} role="img" aria-label="Audio visualization" />

      {/* Drag overlay */}
      <div className={clsx(styles.dropOverlay, { [styles.dropActive]: isDragging })}>
        <div className={styles.dropBorder}>
          <span className={styles.dropText}>Drop audio file here</span>
        </div>
      </div>

      {/* Hint */}
      <div className={clsx(styles.hint, { [styles.hidden]: !!sourceName })}>
        Drop an audio file or click Use Mic
      </div>

      {/* Source indicator */}
      {sourceName && <div className={styles.sourceIndicator}>{sourceName}</div>}

      {/* Mic error */}
      {micError && <div className={styles.sourceIndicator} style={{ color: '#ff6b6b' }}>{micError}</div>}

      {/* Panel toggle */}
      <button
        className={styles.panelToggle}
        onClick={() => setIsPanelOpen(prev => !prev)}
        aria-label={isPanelOpen ? 'Hide controls' : 'Show controls'}
      >
        {isPanelOpen ? '\u2715' : '\u2699'}
      </button>

      {/* Control panel */}
      <div className={clsx(styles.panel, { [styles.panelClosed]: !isPanelOpen })}>
        {/* Audio source */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Source</span>
          <div className={styles.sourceButtons}>
            <button
              className={clsx(styles.sourceButton, {
                [styles.activeSource]: sourceName && sourceName !== 'Microphone'
              })}
              onClick={handleFileClick}
            >
              Drop Audio
            </button>
            <button
              className={clsx(styles.sourceButton, {
                [styles.activeSource]: sourceName === 'Microphone'
              })}
              onClick={handleMic}
            >
              Use Mic
            </button>
          </div>
        </div>

        <div className={styles.panelDivider} />

        {/* Visualization mode */}
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

        <div className={styles.panelDivider} />

        {/* Bloom */}
        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Bloom</span>
          <input
            type="range"
            min="0"
            max="4"
            step="0.1"
            value={bloomStrength}
            onChange={e => setBloomStrength(parseFloat(e.target.value))}
            className={styles.slider}
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

        <div className={styles.panelDivider} />

        {/* Stop */}
        <div className={styles.controlGroup}>
          <button
            className={clsx(styles.actionButton, styles.stopButton)}
            onClick={handleStop}
            disabled={!sourceName}
          >
            Stop
          </button>
        </div>
      </div>
    </div>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => (
  <Seo
    title="Audio Visualizer"
    description="Real-time 3D audio visualization with terrain, radial, and waveform modes - supports file input and microphone"
    heroImage=""
    pathname={pathname}
  />
)

export default AudioVisualizerPage
