export type RulePreset = 'life' | 'highlife' | 'seeds'

interface RuleMask {
  survive: boolean[]
  birth: boolean[]
}

type GenerationListener = (generation: number) => void

const RULE_CONFIG: Record<RulePreset, { survive: number[]; birth: number[] }> = {
  life: { survive: [2, 3], birth: [3] },
  highlife: { survive: [2, 3], birth: [3, 6] },
  seeds: { survive: [], birth: [2] }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function toRuleMask(rule: RulePreset): RuleMask {
  const survive = Array.from({ length: 9 }, () => false)
  const birth = Array.from({ length: 9 }, () => false)
  for (const n of RULE_CONFIG[rule].survive) survive[n] = true
  for (const n of RULE_CONFIG[rule].birth) birth[n] = true
  return { survive, birth }
}

export class CellularAutomataEngine {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D

  private grid = new Uint8Array(0)
  private next = new Uint8Array(0)
  private age = new Uint16Array(0)
  private cols = 0
  private rows = 0

  private w = 1
  private h = 1
  private dpr = 1

  private cellSize = 8
  private wrapEdges = true
  private running = true
  private generation = 0
  private tickInterval = 1000 / 14
  private accumulator = 0

  private rule: RulePreset = 'life'
  private ruleMask: RuleMask = toRuleMask('life')
  private animationId = 0
  private lastTs = 0
  private disposed = false

  private pointerDown = false
  private pointerId: number | null = null
  private paintValue: 0 | 1 = 1

  private resizeObserver: ResizeObserver | null = null
  private dirty = true
  private generationListener: GenerationListener | null = null

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

    this.resizeCanvas(true)
    this.randomize(0.22)
    this.initListeners()
    this.requestFrame()
  }

  setRule(rule: RulePreset) {
    if (this.rule === rule) return
    this.rule = rule
    this.ruleMask = toRuleMask(rule)
  }

  setCellSize(size: number) {
    const nextSize = Math.round(clamp(size, 4, 22))
    if (nextSize === this.cellSize) return
    this.cellSize = nextSize
    this.resizeCanvas(false)
  }

  setSpeed(fps: number) {
    const next = 1000 / clamp(fps, 1, 60)
    if (this.tickInterval === next) return
    this.tickInterval = next
  }

  setRunning(running: boolean) {
    if (this.running === running) return
    this.running = running
    this.lastTs = 0
    if (running) this.requestFrame()
  }

  setWrapEdges(enabled: boolean) {
    if (this.wrapEdges === enabled) return
    this.wrapEdges = enabled
  }

  randomize(density: number) {
    const p = clamp(density, 0, 1)
    for (let i = 0; i < this.grid.length; i++) {
      const alive = Math.random() < p ? 1 : 0
      this.grid[i] = alive
      this.next[i] = 0
      this.age[i] = alive === 1 ? 1 : 0
    }
    this.generation = 0
    this.notifyGeneration()
    this.markDirty()
  }

  clear() {
    this.grid.fill(0)
    this.next.fill(0)
    this.age.fill(0)
    this.generation = 0
    this.notifyGeneration()
    this.markDirty()
  }

  step() {
    this.stepSimulation()
    this.notifyGeneration()
    this.markDirty()
  }

  getGeneration(): number {
    return this.generation
  }

  setGenerationListener(listener: GenerationListener | null) {
    this.generationListener = listener
    this.notifyGeneration()
  }

  dispose() {
    this.disposed = true
    this.generationListener = null
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

  private animate = (ts: number) => {
    if (this.disposed) return
    this.animationId = 0

    let generationChanged = false
    if (this.running) {
      if (this.lastTs === 0) this.lastTs = ts
      const dt = Math.min(100, ts - this.lastTs)
      this.lastTs = ts

      this.accumulator += dt
      let stepped = false
      while (this.accumulator >= this.tickInterval) {
        this.stepSimulation()
        this.accumulator -= this.tickInterval
        stepped = true
      }
      if (stepped) {
        generationChanged = true
        this.dirty = true
      }
    } else {
      this.lastTs = 0
      this.accumulator = 0
    }

    if (generationChanged) this.notifyGeneration()

    if (this.dirty) {
      this.draw()
      this.dirty = false
    }

    if (this.running || this.dirty) this.requestFrame()
  }

  private stepSimulation() {
    if (this.cols === 0 || this.rows === 0) return

    const source = this.grid
    const target = this.next

    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < this.cols; col++) {
        const idx = row * this.cols + col
        const neighbors = this.countNeighbors(source, col, row)
        const isAlive = source[idx] === 1
        target[idx] = (isAlive ? this.ruleMask.survive[neighbors] : this.ruleMask.birth[neighbors]) ? 1 : 0
      }
    }

    this.grid = target
    this.next = source

    for (let i = 0; i < this.grid.length; i++) {
      if (this.grid[i] === 1) {
        const prevAge = this.age[i]
        this.age[i] = prevAge < 65535 ? prevAge + 1 : prevAge
      } else {
        this.age[i] = 0
      }
    }

    this.generation++
  }

  private countNeighbors(source: Uint8Array, col: number, row: number): number {
    let count = 0

    for (let oy = -1; oy <= 1; oy++) {
      for (let ox = -1; ox <= 1; ox++) {
        if (ox === 0 && oy === 0) continue

        let nx = col + ox
        let ny = row + oy

        if (this.wrapEdges) {
          nx = (nx + this.cols) % this.cols
          ny = (ny + this.rows) % this.rows
        } else if (nx < 0 || nx >= this.cols || ny < 0 || ny >= this.rows) {
          continue
        }

        if (source[ny * this.cols + nx] === 1) {
          count++
        }
      }
    }

    return count
  }

  private draw() {
    this.ctx.clearRect(0, 0, this.w, this.h)
    this.ctx.fillStyle = '#040812'
    this.ctx.fillRect(0, 0, this.w, this.h)

    const step = this.cellSize
    const cellDrawSize = Math.max(1, step - 1)

    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < this.cols; col++) {
        const idx = row * this.cols + col
        if (this.grid[idx] === 0) continue

        const age = this.age[idx]
        const hue = (185 + age * 2.1 + col * 0.45 + row * 0.35) % 360
        const light = Math.min(74, 40 + age * 1.1)
        this.ctx.fillStyle = `hsl(${hue} 86% ${light}%)`
        this.ctx.fillRect(col * step + 0.5, row * step + 0.5, cellDrawSize, cellDrawSize)
      }
    }

    if (this.cellSize >= 12) {
      this.ctx.strokeStyle = 'rgba(124, 162, 255, 0.08)'
      this.ctx.lineWidth = 1
      this.ctx.beginPath()

      for (let x = 0; x <= this.cols; x++) {
        const px = x * this.cellSize + 0.5
        this.ctx.moveTo(px, 0)
        this.ctx.lineTo(px, this.h)
      }
      for (let y = 0; y <= this.rows; y++) {
        const py = y * this.cellSize + 0.5
        this.ctx.moveTo(0, py)
        this.ctx.lineTo(this.w, py)
      }
      this.ctx.stroke()
    }
  }

  private resizeCanvas(initial: boolean) {
    const rect = this.container.getBoundingClientRect()
    this.w = Math.max(1, Math.floor(rect.width))
    this.h = Math.max(1, Math.floor(rect.height))
    this.dpr = Math.min(window.devicePixelRatio || 1, 2)

    this.canvas.width = Math.floor(this.w * this.dpr)
    this.canvas.height = Math.floor(this.h * this.dpr)
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0)
    this.ctx.imageSmoothingEnabled = false

    this.rebuildGrid(!initial)
  }

  private rebuildGrid(preserve: boolean) {
    const newCols = Math.max(8, Math.floor(this.w / this.cellSize))
    const newRows = Math.max(8, Math.floor(this.h / this.cellSize))
    const newLength = newCols * newRows
    const newGrid = new Uint8Array(newLength)
    const newAge = new Uint16Array(newLength)

    if (preserve && this.cols > 0 && this.rows > 0) {
      const copyCols = Math.min(this.cols, newCols)
      const copyRows = Math.min(this.rows, newRows)
      const srcX = Math.floor((this.cols - copyCols) / 2)
      const srcY = Math.floor((this.rows - copyRows) / 2)
      const dstX = Math.floor((newCols - copyCols) / 2)
      const dstY = Math.floor((newRows - copyRows) / 2)

      for (let row = 0; row < copyRows; row++) {
        for (let col = 0; col < copyCols; col++) {
          const srcIdx = (srcY + row) * this.cols + (srcX + col)
          const dstIdx = (dstY + row) * newCols + (dstX + col)
          newGrid[dstIdx] = this.grid[srcIdx]
          newAge[dstIdx] = this.age[srcIdx]
        }
      }
    }

    this.cols = newCols
    this.rows = newRows
    this.grid = newGrid
    this.next = new Uint8Array(newLength)
    this.age = newAge
    this.accumulator = 0
    this.markDirty()
  }

  private markDirty() {
    this.dirty = true
    this.requestFrame()
  }

  private requestFrame() {
    if (this.disposed || this.animationId !== 0) return
    this.animationId = requestAnimationFrame(this.animate)
  }

  private notifyGeneration() {
    if (!this.generationListener) return
    this.generationListener(this.generation)
  }

  private initListeners() {
    this.resizeObserver = new ResizeObserver(() => this.resizeCanvas(false))
    this.resizeObserver.observe(this.container)

    this.canvas.addEventListener('pointerdown', this.onPointerDown)
    window.addEventListener('pointermove', this.onPointerMove)
    window.addEventListener('pointerup', this.onPointerUp)
    window.addEventListener('pointercancel', this.onPointerUp)
    this.canvas.addEventListener('contextmenu', this.onContextMenu)
  }

  private onPointerDown = (e: PointerEvent) => {
    if (e.button !== 0) return

    const idx = this.indexFromClient(e.clientX, e.clientY)
    if (idx < 0) return

    this.pointerDown = true
    this.pointerId = e.pointerId
    this.paintValue = this.grid[idx] === 1 ? 0 : 1

    this.paintCell(idx, this.paintValue)
    this.canvas.setPointerCapture(e.pointerId)
  }

  private onPointerMove = (e: PointerEvent) => {
    if (!this.pointerDown || this.pointerId !== e.pointerId) return
    const idx = this.indexFromClient(e.clientX, e.clientY)
    if (idx < 0) return
    this.paintCell(idx, this.paintValue)
  }

  private onPointerUp = (e: PointerEvent) => {
    if (this.pointerId !== e.pointerId) return
    this.pointerDown = false
    this.pointerId = null
    if (this.canvas.hasPointerCapture(e.pointerId)) {
      this.canvas.releasePointerCapture(e.pointerId)
    }
  }

  private onContextMenu = (e: MouseEvent) => {
    e.preventDefault()
  }

  private paintCell(index: number, value: 0 | 1) {
    if (this.grid[index] === value) return
    this.grid[index] = value
    this.age[index] = value === 1 ? Math.max(1, this.age[index]) : 0
    this.markDirty()
  }

  private indexFromClient(clientX: number, clientY: number): number {
    const rect = this.canvas.getBoundingClientRect()
    const rx = clientX - rect.left
    const ry = clientY - rect.top
    if (rx < 0 || ry < 0 || rx >= rect.width || ry >= rect.height) return -1

    const x = (rx / rect.width) * this.w
    const y = (ry / rect.height) * this.h
    const col = Math.floor(x / this.cellSize)
    const row = Math.floor(y / this.cellSize)

    if (col < 0 || col >= this.cols || row < 0 || row >= this.rows) return -1
    return row * this.cols + col
  }
}
