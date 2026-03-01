export type WavePalette = 'neon' | 'sunset' | 'mono'

interface WaveSource {
  x: number
  y: number
  phase: number
  speed: number
  strength: number
}

type SourceCountListener = (count: number) => void

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function clampColor(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)))
}

export class WaveInterferenceEngine {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D

  private bufferCanvas: HTMLCanvasElement
  private bufferCtx: CanvasRenderingContext2D
  private imageData: ImageData | null = null
  private pixels: Uint8ClampedArray | null = null

  private w = 1
  private h = 1
  private dpr = 1
  private lowW = 1
  private lowH = 1

  private resolution = 4
  private frequency = 1.2
  private wavelength = 34
  private damping = 0.008
  private amplitude = 1
  private palette: WavePalette = 'neon'
  private running = true

  private sources: WaveSource[] = []
  private time = 0
  private lastTs = 0
  private animationId = 0
  private disposed = false

  private dragSourceIndex: number | null = null
  private pointerId: number | null = null
  private resizeObserver: ResizeObserver | null = null
  private dirty = true
  private sourceCountListener: SourceCountListener | null = null

  private readonly maxSources = 8
  private readonly paletteSteps = 1024
  private paletteLut = new Uint8ClampedArray(this.paletteSteps * 3)
  private attenuationLut = new Float32Array(0)
  private attenuationMaxDist = 0
  private sourceX = new Float32Array(this.maxSources)
  private sourceY = new Float32Array(this.maxSources)
  private sourcePhase = new Float32Array(this.maxSources)
  private sourceStrength = new Float32Array(this.maxSources)

  constructor(private container: HTMLDivElement) {
    this.canvas = document.createElement('canvas')
    this.canvas.style.width = '100%'
    this.canvas.style.height = '100%'
    this.canvas.style.display = 'block'
    this.canvas.style.touchAction = 'none'
    container.appendChild(this.canvas)

    const ctx = this.canvas.getContext('2d')
    if (!ctx) {
      throw new Error('Canvas 2D context is not available')
    }
    this.ctx = ctx

    this.bufferCanvas = document.createElement('canvas')
    const bufferCtx = this.bufferCanvas.getContext('2d')
    if (!bufferCtx) {
      throw new Error('Offscreen 2D context is not available')
    }
    this.bufferCtx = bufferCtx

    this.resizeCanvas()
    this.rebuildPaletteLut()
    this.resetSources()
    this.initListeners()
    this.requestFrame()
  }

  setRunning(running: boolean) {
    if (this.running === running) return
    this.running = running
    if (!running) this.lastTs = 0
    this.markDirty()
  }

  setFrequency(value: number) {
    const next = clamp(value, 0.2, 4)
    if (this.frequency === next) return
    this.frequency = next
    this.markDirty()
  }

  setWavelength(value: number) {
    const next = clamp(value, 8, 160)
    if (this.wavelength === next) return
    this.wavelength = next
    this.markDirty()
  }

  setDamping(value: number) {
    const next = clamp(value, 0, 0.04)
    if (this.damping === next) return
    this.damping = next
    this.rebuildAttenuationLut()
    this.markDirty()
  }

  setAmplitude(value: number) {
    const next = clamp(value, 0.2, 3)
    if (this.amplitude === next) return
    this.amplitude = next
    this.markDirty()
  }

  setResolution(value: number) {
    const nextResolution = Math.round(clamp(value, 2, 12))
    if (nextResolution === this.resolution) return
    this.resolution = nextResolution
    this.resizeBuffers()
    this.markDirty()
  }

  setPalette(palette: WavePalette) {
    if (this.palette === palette) return
    this.palette = palette
    this.rebuildPaletteLut()
    this.markDirty()
  }

  setSourceCount(count: number) {
    const target = Math.round(clamp(count, 1, this.maxSources))
    if (target === this.sources.length) return
    if (target < this.sources.length) {
      this.sources.length = target
      if (this.dragSourceIndex !== null && this.dragSourceIndex >= target) {
        this.dragSourceIndex = null
      }
      this.notifySourceCountChanged()
      this.markDirty()
      return
    }
    while (this.sources.length < target) {
      this.sources.push(this.createSource(Math.random() * this.w, Math.random() * this.h))
    }
    this.notifySourceCountChanged()
    this.markDirty()
  }

  getSourceCount(): number {
    return this.sources.length
  }

  setSourceCountListener(listener: SourceCountListener | null) {
    this.sourceCountListener = listener
    this.notifySourceCountChanged()
  }

  resetSources() {
    const cx = this.w * 0.5
    const cy = this.h * 0.5
    const radius = Math.max(40, Math.min(this.w, this.h) * 0.24)
    this.sources = [
      this.createSource(cx - radius, cy + radius * 0.15),
      this.createSource(cx + radius, cy + radius * 0.15),
      this.createSource(cx, cy - radius * 0.85)
    ]
    this.notifySourceCountChanged()
    this.markDirty()
  }

  dispose() {
    this.disposed = true
    this.sourceCountListener = null
    if (this.animationId !== 0) {
      cancelAnimationFrame(this.animationId)
      this.animationId = 0
    }

    this.resizeObserver?.disconnect()
    this.resizeObserver = null

    this.canvas.removeEventListener('pointerdown', this.onPointerDown)
    window.removeEventListener('pointermove', this.onPointerMove)
    window.removeEventListener('pointerup', this.onPointerUp)
    window.removeEventListener('pointercancel', this.onPointerUp)
    this.canvas.removeEventListener('contextmenu', this.onContextMenu)

    if (this.canvas.parentElement) {
      this.canvas.parentElement.removeChild(this.canvas)
    }
  }

  private createSource(x: number, y: number): WaveSource {
    return {
      x: clamp(x, 0, this.w),
      y: clamp(y, 0, this.h),
      phase: Math.random() * Math.PI * 2,
      speed: 0.6 + Math.random() * 1.4,
      strength: 0.85 + Math.random() * 0.45
    }
  }

  private animate = (ts: number) => {
    if (this.disposed) return
    this.animationId = 0

    if (this.lastTs === 0) this.lastTs = ts
    const dt = Math.min(0.05, (ts - this.lastTs) / 1000)
    this.lastTs = ts

    if (this.running) {
      this.time += dt * this.frequency * Math.PI * 2
      for (const source of this.sources) {
        source.phase += dt * source.speed * this.frequency * 2.3
      }
      this.dirty = true
    }

    if (this.dirty) {
      this.renderField()
      this.dirty = false
    }

    if (this.running || this.dirty) {
      this.requestFrame()
    } else {
      this.lastTs = 0
    }
  }

  private renderField() {
    if (!this.imageData || !this.pixels) return

    const count = this.sources.length
    const invCount = count > 0 ? 1 / count : 0
    const invWavelength = 1 / this.wavelength
    const attenuationLut = this.attenuationLut
    const attenuationMaxDist = this.attenuationMaxDist
    const paletteLut = this.paletteLut
    const paletteMaxIndex = this.paletteSteps - 1

    for (let i = 0; i < count; i++) {
      const source = this.sources[i]
      this.sourceX[i] = source.x
      this.sourceY[i] = source.y
      this.sourcePhase[i] = source.phase - this.time
      this.sourceStrength[i] = source.strength * this.amplitude
    }

    let p = 0

    for (let y = 0; y < this.lowH; y++) {
      const wy = ((y + 0.5) / this.lowH) * this.h
      for (let x = 0; x < this.lowW; x++) {
        const wx = ((x + 0.5) / this.lowW) * this.w

        let sum = 0
        for (let i = 0; i < count; i++) {
          const dx = wx - this.sourceX[i]
          const dy = wy - this.sourceY[i]
          const dist = Math.sqrt(dx * dx + dy * dy)
          const attenuation = attenuationLut[Math.min(attenuationMaxDist, dist | 0)]
          const wave = Math.sin(dist * invWavelength + this.sourcePhase[i])
          sum += wave * attenuation * this.sourceStrength[i]
        }

        const normalized = clamp(sum * invCount, -1, 1)
        const lutIndex = Math.round(((normalized + 1) * 0.5) * paletteMaxIndex)
        const lutBase = lutIndex * 3
        this.pixels[p++] = paletteLut[lutBase]
        this.pixels[p++] = paletteLut[lutBase + 1]
        this.pixels[p++] = paletteLut[lutBase + 2]
        this.pixels[p++] = 255
      }
    }

    this.bufferCtx.putImageData(this.imageData, 0, 0)
    this.ctx.imageSmoothingEnabled = false
    this.ctx.drawImage(this.bufferCanvas, 0, 0, this.w, this.h)
    this.drawSources()
  }

  private samplePalette(v: number): [number, number, number] {
    const t = (v + 1) * 0.5

    if (this.palette === 'sunset') {
      const r = clampColor(80 + 175 * t)
      const g = clampColor(25 + 155 * Math.pow(1 - Math.abs(t - 0.5) * 2, 0.85))
      const b = clampColor(28 + 130 * (1 - t))
      return [r, g, b]
    }

    if (this.palette === 'mono') {
      const l = clampColor(22 + 210 * t)
      return [clampColor(l * 0.9), clampColor(l * 0.95), l]
    }

    const r = clampColor(30 + 220 * Math.pow(t, 1.1))
    const g = clampColor(10 + 210 * Math.pow(1 - Math.abs(t - 0.5) * 2, 0.75))
    const b = clampColor(75 + 170 * Math.pow(1 - t, 0.8))
    return [r, g, b]
  }

  private drawSources() {
    for (let i = 0; i < this.sources.length; i++) {
      const source = this.sources[i]
      const pulse = 5 + Math.sin(this.time * 2 + i) * 1.5

      this.ctx.beginPath()
      this.ctx.arc(source.x, source.y, pulse + 8, 0, Math.PI * 2)
      this.ctx.fillStyle = 'rgba(255, 255, 255, 0.08)'
      this.ctx.fill()

      this.ctx.beginPath()
      this.ctx.arc(source.x, source.y, pulse, 0, Math.PI * 2)
      this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
      this.ctx.fill()

      this.ctx.beginPath()
      this.ctx.arc(source.x, source.y, pulse + 12, 0, Math.PI * 2)
      this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.26)'
      this.ctx.lineWidth = 1.2
      this.ctx.stroke()
    }
  }

  private resizeCanvas() {
    const rect = this.container.getBoundingClientRect()
    this.w = Math.max(1, Math.floor(rect.width))
    this.h = Math.max(1, Math.floor(rect.height))
    this.dpr = Math.min(window.devicePixelRatio || 1, 2)

    this.canvas.width = Math.floor(this.w * this.dpr)
    this.canvas.height = Math.floor(this.h * this.dpr)
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0)
    this.ctx.imageSmoothingEnabled = false

    this.resizeBuffers()
    this.rebuildAttenuationLut()
    for (const source of this.sources) {
      source.x = clamp(source.x, 0, this.w)
      source.y = clamp(source.y, 0, this.h)
    }
    this.markDirty()
  }

  private resizeBuffers() {
    this.lowW = Math.max(12, Math.ceil(this.w / this.resolution))
    this.lowH = Math.max(12, Math.ceil(this.h / this.resolution))
    this.bufferCanvas.width = this.lowW
    this.bufferCanvas.height = this.lowH
    this.imageData = this.bufferCtx.createImageData(this.lowW, this.lowH)
    this.pixels = this.imageData.data
  }

  private rebuildPaletteLut() {
    for (let i = 0; i < this.paletteSteps; i++) {
      const v = (i / (this.paletteSteps - 1)) * 2 - 1
      const [r, g, b] = this.samplePalette(v)
      const base = i * 3
      this.paletteLut[base] = r
      this.paletteLut[base + 1] = g
      this.paletteLut[base + 2] = b
    }
  }

  private rebuildAttenuationLut() {
    this.attenuationMaxDist = Math.ceil(Math.sqrt(this.w * this.w + this.h * this.h)) + 2
    this.attenuationLut = new Float32Array(this.attenuationMaxDist + 1)
    for (let i = 0; i <= this.attenuationMaxDist; i++) {
      this.attenuationLut[i] = Math.exp(-i * this.damping)
    }
  }

  private markDirty() {
    this.dirty = true
    this.requestFrame()
  }

  private requestFrame() {
    if (this.disposed || this.animationId !== 0) return
    this.animationId = requestAnimationFrame(this.animate)
  }

  private notifySourceCountChanged() {
    if (!this.sourceCountListener) return
    this.sourceCountListener(this.sources.length)
  }

  private initListeners() {
    this.resizeObserver = new ResizeObserver(() => this.resizeCanvas())
    this.resizeObserver.observe(this.container)

    this.canvas.addEventListener('pointerdown', this.onPointerDown)
    window.addEventListener('pointermove', this.onPointerMove)
    window.addEventListener('pointerup', this.onPointerUp)
    window.addEventListener('pointercancel', this.onPointerUp)
    this.canvas.addEventListener('contextmenu', this.onContextMenu)
  }

  private onPointerDown = (e: PointerEvent) => {
    if (e.button !== 0) return

    const pos = this.clientToWorld(e.clientX, e.clientY)
    if (!pos) return

    const nearest = this.findNearestSource(pos.x, pos.y, 26)
    if (nearest >= 0) {
      this.dragSourceIndex = nearest
    } else if (this.sources.length < this.maxSources) {
      this.sources.push(this.createSource(pos.x, pos.y))
      this.dragSourceIndex = this.sources.length - 1
      this.notifySourceCountChanged()
      this.markDirty()
    } else {
      this.dragSourceIndex = this.findNearestSource(pos.x, pos.y, Infinity)
    }

    this.pointerId = e.pointerId
    this.canvas.setPointerCapture(e.pointerId)
  }

  private onPointerMove = (e: PointerEvent) => {
    if (this.pointerId !== e.pointerId || this.dragSourceIndex === null) return
    const pos = this.clientToWorld(e.clientX, e.clientY)
    if (!pos) return

    const source = this.sources[this.dragSourceIndex]
    if (!source) return
    source.x = clamp(pos.x, 0, this.w)
    source.y = clamp(pos.y, 0, this.h)
    this.markDirty()
  }

  private onPointerUp = (e: PointerEvent) => {
    if (this.pointerId !== e.pointerId) return
    this.dragSourceIndex = null
    this.pointerId = null
    if (this.canvas.hasPointerCapture(e.pointerId)) {
      this.canvas.releasePointerCapture(e.pointerId)
    }
    this.markDirty()
  }

  private onContextMenu = (e: MouseEvent) => {
    e.preventDefault()
    if (this.sources.length <= 1) return

    const pos = this.clientToWorld(e.clientX, e.clientY)
    if (!pos) return
    const nearest = this.findNearestSource(pos.x, pos.y, 30)
    if (nearest >= 0) {
      this.sources.splice(nearest, 1)
      if (this.dragSourceIndex !== null && this.dragSourceIndex >= this.sources.length) {
        this.dragSourceIndex = null
      }
      this.notifySourceCountChanged()
      this.markDirty()
    }
  }

  private clientToWorld(clientX: number, clientY: number): { x: number; y: number } | null {
    const rect = this.canvas.getBoundingClientRect()
    const rx = clientX - rect.left
    const ry = clientY - rect.top
    if (rx < 0 || ry < 0 || rx >= rect.width || ry >= rect.height) return null
    return {
      x: (rx / rect.width) * this.w,
      y: (ry / rect.height) * this.h
    }
  }

  private findNearestSource(x: number, y: number, maxDistance: number): number {
    const maxDistSq = maxDistance * maxDistance
    let best = maxDistSq
    let nearest = -1

    for (let i = 0; i < this.sources.length; i++) {
      const source = this.sources[i]
      const dx = source.x - x
      const dy = source.y - y
      const distSq = dx * dx + dy * dy
      if (distSq <= best) {
        best = distSq
        nearest = i
      }
    }

    return nearest
  }
}
