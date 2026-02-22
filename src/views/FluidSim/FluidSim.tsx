import { useRef, useEffect, useState, useCallback } from 'react'
import type { HeadProps } from 'gatsby'
import clsx from 'clsx'
import { Seo } from '@/components'

import * as styles from './FluidSim.module.scss'
import type { FluidConfig } from './fluid'

type FluidSimulation = import('./fluid').FluidSimulation

const FluidSimPage = () => {
  const canvasRef = useRef<HTMLDivElement>(null)
  const simRef = useRef<FluidSimulation | null>(null)
  const [isSSR, setIsSSR] = useState(true)
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [config, setConfig] = useState<FluidConfig>({
    viscosity: 0.3,
    curl: 30,
    pressure: 30,
    splatRadius: 0.25
  })

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    if (isSSR || !canvasRef.current) return

    let sim: FluidSimulation | null = null
    let cancelled = false

    import('./fluid').then(({ FluidSimulation }) => {
      if (cancelled || !canvasRef.current) return
      sim = new FluidSimulation(canvasRef.current)
      simRef.current = sim
    })

    return () => {
      cancelled = true
      if (sim) {
        sim.dispose()
        simRef.current = null
      }
    }
  }, [isSSR])

  useEffect(() => {
    if (simRef.current) simRef.current.setConfig(config)
  }, [config])

  const updateConfig = useCallback((key: keyof FluidConfig, value: number) => {
    setConfig(prev => ({ ...prev, [key]: value }))
  }, [])

  const resettingRef = useRef(false)
  const handleReset = useCallback(() => {
    if (!simRef.current || !canvasRef.current || resettingRef.current) return
    resettingRef.current = true
    simRef.current.dispose()
    simRef.current = null
    import('./fluid').then(({ FluidSimulation }) => {
      resettingRef.current = false
      if (!canvasRef.current) return
      const sim = new FluidSimulation(canvasRef.current)
      sim.setConfig(config)
      simRef.current = sim
    })
  }, [config])

  if (isSSR) {
    return <div className={styles.page} />
  }

  return (
    <div className={styles.page}>
      <div ref={canvasRef} className={styles.canvas} />

      <div className={styles.hint}>
        Draw with your mouse to create fluid
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
          <span className={styles.controlLabel}>Viscosity</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={config.viscosity}
            onChange={e => updateConfig('viscosity', parseFloat(e.target.value))}
            className={styles.slider}
          />
        </div>

        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Curl</span>
          <input
            type="range"
            min="0"
            max="60"
            step="1"
            value={config.curl}
            onChange={e => updateConfig('curl', parseFloat(e.target.value))}
            className={styles.slider}
          />
        </div>

        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Pressure</span>
          <input
            type="range"
            min="5"
            max="60"
            step="1"
            value={config.pressure}
            onChange={e => updateConfig('pressure', parseFloat(e.target.value))}
            className={styles.slider}
          />
        </div>

        <div className={styles.controlGroup}>
          <span className={styles.controlLabel}>Splat Size</span>
          <input
            type="range"
            min="0.05"
            max="0.8"
            step="0.01"
            value={config.splatRadius}
            onChange={e => updateConfig('splatRadius', parseFloat(e.target.value))}
            className={styles.slider}
          />
        </div>

        <div className={styles.panelDivider} />

        <div className={styles.controlGroup}>
          <button className={styles.actionButton} onClick={handleReset}>
            Reset
          </button>
        </div>
      </div>
    </div>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => (
  <Seo
    title="Fluid Simulation"
    description="GPU-accelerated fluid dynamics — paint with colorful ink, watch vortices and smoke swirl in real-time"
    heroImage=""
    pathname={pathname}
  />
)

export default FluidSimPage
