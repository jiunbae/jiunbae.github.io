export type SortAlgorithm = 'quick' | 'bubble' | 'selection' | 'insertion'
export type Distribution = 'random' | 'reversed' | 'nearlySorted'

interface CompareOperation {
  type: 'compare'
  a: number
  b: number
}

interface SwapOperation {
  type: 'swap'
  a: number
  b: number
}

type SortOperation = CompareOperation | SwapOperation

export interface SortSnapshot {
  running: boolean
  done: boolean
  algorithm: SortAlgorithm
  size: number
  distribution: Distribution
  comparisons: number
  swaps: number
  operations: number
  totalOperations: number
  progress: number
}

type SnapshotListener = (snapshot: SortSnapshot) => void

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function swap(values: number[], a: number, b: number) {
  const temp = values[a]
  values[a] = values[b]
  values[b] = temp
}

export class SortingLabEngine {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D

  private values: number[] = []
  private algorithm: SortAlgorithm = 'quick'
  private distribution: Distribution = 'random'
  private size = 120
  private opsPerFrame = 80

  private operations: SortOperation[] = []
  private opIndex = 0
  private dirty = true
  private running = false
  private done = false

  private comparisons = 0
  private swaps = 0

  private highlightA = -1
  private highlightB = -1
  private highlightTtl = 0
  private dirtyFrame = true
  private backgroundGradient: CanvasGradient | null = null

  private w = 1
  private h = 1
  private dpr = 1
  private animationId = 0
  private disposed = false
  private resizeObserver: ResizeObserver | null = null
  private snapshotListener: SnapshotListener | null = null
  private snapshotDirty = true

  constructor(private container: HTMLDivElement) {
    this.canvas = document.createElement('canvas')
    this.canvas.style.width = '100%'
    this.canvas.style.height = '100%'
    this.canvas.style.display = 'block'
    container.appendChild(this.canvas)

    const ctx = this.canvas.getContext('2d')
    if (!ctx) {
      throw new Error('Canvas 2D context is not available')
    }
    this.ctx = ctx

    this.resizeCanvas()
    this.shuffle('random')
    this.initListeners()
    this.requestFrame()
  }

  setAlgorithm(algorithm: SortAlgorithm) {
    if (this.algorithm === algorithm) return
    this.algorithm = algorithm
    this.running = false
    this.done = false
    this.markDirty()
  }

  setDistribution(distribution: Distribution) {
    if (this.distribution === distribution) return
    this.shuffle(distribution)
  }

  setSize(size: number) {
    const target = Math.round(clamp(size, 16, 360))
    if (this.size === target) return
    this.size = target
    this.shuffle(this.distribution)
  }

  setSpeed(opsPerFrame: number) {
    const next = Math.round(clamp(opsPerFrame, 1, 400))
    if (this.opsPerFrame === next) return
    this.opsPerFrame = next
  }

  shuffle(distribution: Distribution = this.distribution) {
    this.distribution = distribution

    const values = Array.from({ length: this.size }, (_, i) => i + 1)
    if (distribution === 'random') {
      for (let i = values.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        swap(values, i, j)
      }
    } else if (distribution === 'reversed') {
      values.reverse()
    } else {
      const swaps = Math.max(1, Math.floor(this.size * 0.08))
      for (let i = 0; i < swaps; i++) {
        const a = Math.floor(Math.random() * values.length)
        const b = Math.floor(Math.random() * values.length)
        swap(values, a, b)
      }
    }

    this.values = values
    this.running = false
    this.done = false
    this.markDirty()
  }

  start() {
    this.ensureOperations()
    if (this.done) return
    this.running = true
    this.markSnapshotDirty()
    this.markFrameDirty()
  }

  pause() {
    if (!this.running) return
    this.running = false
    this.markSnapshotDirty()
    this.markFrameDirty()
  }

  step(steps = 1) {
    this.ensureOperations()
    if (this.done) return
    this.running = false
    for (let i = 0; i < steps; i++) {
      this.applyNextOperation()
      if (this.done) break
    }
    this.markSnapshotDirty()
    this.markFrameDirty()
  }

  getSnapshot(): SortSnapshot {
    const total = this.operations.length
    const progress = this.done ? 1 : total > 0 ? this.opIndex / total : 0
    return {
      running: this.running,
      done: this.done,
      algorithm: this.algorithm,
      size: this.size,
      distribution: this.distribution,
      comparisons: this.comparisons,
      swaps: this.swaps,
      operations: this.opIndex,
      totalOperations: total,
      progress
    }
  }

  setSnapshotListener(listener: SnapshotListener | null) {
    this.snapshotListener = listener
    this.snapshotDirty = true
    this.flushSnapshot()
  }

  dispose() {
    this.disposed = true
    this.snapshotListener = null
    if (this.animationId !== 0) {
      cancelAnimationFrame(this.animationId)
      this.animationId = 0
    }
    this.resizeObserver?.disconnect()
    this.resizeObserver = null

    if (this.canvas.parentElement) {
      this.canvas.parentElement.removeChild(this.canvas)
    }
  }

  private animate = () => {
    if (this.disposed) return
    this.animationId = 0

    if (this.running) {
      this.ensureOperations()
      for (let i = 0; i < this.opsPerFrame; i++) {
        this.applyNextOperation()
        if (this.done) break
      }
    }

    if (this.highlightTtl > 0) {
      this.highlightTtl--
      if (this.highlightTtl === 0) {
        this.highlightA = -1
        this.highlightB = -1
        this.dirtyFrame = true
      }
    }

    if (this.running || this.dirtyFrame) {
      this.draw()
      this.dirtyFrame = false
    }
    this.flushSnapshot()

    if (this.running || this.highlightTtl > 0 || this.dirtyFrame) {
      this.requestFrame()
    }
  }

  private applyNextOperation() {
    if (this.opIndex >= this.operations.length) {
      this.running = false
      this.done = true
      this.markSnapshotDirty()
      return
    }

    const op = this.operations[this.opIndex++]
    this.highlightA = op.a
    this.highlightB = op.b
    this.highlightTtl = 6
    this.dirtyFrame = true

    if (op.type === 'compare') {
      this.comparisons++
    } else {
      this.swaps++
      swap(this.values, op.a, op.b)
    }
    this.markSnapshotDirty()

    if (this.opIndex >= this.operations.length) {
      this.running = false
      this.done = true
      this.markSnapshotDirty()
    }
  }

  private draw() {
    this.ctx.clearRect(0, 0, this.w, this.h)

    if (!this.backgroundGradient) {
      const bg = this.ctx.createLinearGradient(0, 0, 0, this.h)
      bg.addColorStop(0, '#0f172a')
      bg.addColorStop(0.45, '#111827')
      bg.addColorStop(1, '#030712')
      this.backgroundGradient = bg
    }
    this.ctx.fillStyle = this.backgroundGradient
    this.ctx.fillRect(0, 0, this.w, this.h)

    const n = this.values.length
    if (n === 0) return

    const barWidth = this.w / n
    const gap = barWidth > 4 ? 1 : 0
    const maxHeight = this.h - 18

    for (let i = 0; i < n; i++) {
      const value = this.values[i]
      const t = value / n
      const height = Math.max(3, t * maxHeight)
      const x = i * barWidth + gap * 0.5
      const y = this.h - height
      const width = Math.max(1, barWidth - gap)

      if (this.done) {
        this.ctx.fillStyle = `hsl(${155 + t * 30} 70% ${46 + t * 14}%)`
      } else if (i === this.highlightA) {
        this.ctx.fillStyle = '#fb7185'
      } else if (i === this.highlightB) {
        this.ctx.fillStyle = '#22d3ee'
      } else {
        this.ctx.fillStyle = `hsl(${205 + t * 90} 72% ${38 + t * 20}%)`
      }

      this.ctx.fillRect(x, y, width, height)
    }

    this.ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)'
    this.ctx.lineWidth = 1
    this.ctx.beginPath()
    this.ctx.moveTo(0, this.h - 0.5)
    this.ctx.lineTo(this.w, this.h - 0.5)
    this.ctx.stroke()
  }

  private ensureOperations() {
    if (!this.dirty) return

    const working = [...this.values]
    this.operations = this.generateOperations(working, this.algorithm)
    this.opIndex = 0
    this.comparisons = 0
    this.swaps = 0
    this.running = false
    this.done = this.operations.length === 0
    this.highlightA = -1
    this.highlightB = -1
    this.highlightTtl = 0
    this.dirty = false
    this.markSnapshotDirty()
    this.markFrameDirty()
  }

  private generateOperations(values: number[], algorithm: SortAlgorithm): SortOperation[] {
    if (algorithm === 'bubble') return this.buildBubbleOperations(values)
    if (algorithm === 'selection') return this.buildSelectionOperations(values)
    if (algorithm === 'insertion') return this.buildInsertionOperations(values)
    return this.buildQuickOperations(values)
  }

  private buildBubbleOperations(values: number[]): SortOperation[] {
    const ops: SortOperation[] = []
    const n = values.length

    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < n - i - 1; j++) {
        ops.push({ type: 'compare', a: j, b: j + 1 })
        if (values[j] > values[j + 1]) {
          swap(values, j, j + 1)
          ops.push({ type: 'swap', a: j, b: j + 1 })
        }
      }
    }

    return ops
  }

  private buildSelectionOperations(values: number[]): SortOperation[] {
    const ops: SortOperation[] = []
    const n = values.length

    for (let i = 0; i < n - 1; i++) {
      let minIndex = i
      for (let j = i + 1; j < n; j++) {
        ops.push({ type: 'compare', a: minIndex, b: j })
        if (values[j] < values[minIndex]) {
          minIndex = j
        }
      }
      if (minIndex !== i) {
        swap(values, i, minIndex)
        ops.push({ type: 'swap', a: i, b: minIndex })
      }
    }

    return ops
  }

  private buildInsertionOperations(values: number[]): SortOperation[] {
    const ops: SortOperation[] = []
    const n = values.length

    for (let i = 1; i < n; i++) {
      let j = i
      while (j > 0) {
        ops.push({ type: 'compare', a: j - 1, b: j })
        if (values[j - 1] <= values[j]) break
        swap(values, j - 1, j)
        ops.push({ type: 'swap', a: j - 1, b: j })
        j--
      }
    }

    return ops
  }

  private buildQuickOperations(values: number[]): SortOperation[] {
    const ops: SortOperation[] = []
    const stack: Array<[number, number]> = [[0, values.length - 1]]

    while (stack.length > 0) {
      const range = stack.pop()
      if (!range) continue
      const [low, high] = range
      if (low >= high) continue

      const pivotIndex = this.partition(values, low, high, ops)
      const leftSize = pivotIndex - 1 - low
      const rightSize = high - (pivotIndex + 1)

      if (leftSize > rightSize) {
        stack.push([low, pivotIndex - 1])
        stack.push([pivotIndex + 1, high])
      } else {
        stack.push([pivotIndex + 1, high])
        stack.push([low, pivotIndex - 1])
      }
    }

    return ops
  }

  private partition(values: number[], low: number, high: number, ops: SortOperation[]): number {
    const pivot = values[high]
    let i = low

    for (let j = low; j < high; j++) {
      ops.push({ type: 'compare', a: j, b: high })
      if (values[j] < pivot) {
        if (i !== j) {
          swap(values, i, j)
          ops.push({ type: 'swap', a: i, b: j })
        }
        i++
      }
    }

    if (i !== high) {
      swap(values, i, high)
      ops.push({ type: 'swap', a: i, b: high })
    }

    return i
  }

  private markDirty() {
    this.dirty = true
    this.operations = []
    this.opIndex = 0
    this.comparisons = 0
    this.swaps = 0
    this.highlightA = -1
    this.highlightB = -1
    this.highlightTtl = 0
    this.markSnapshotDirty()
    this.markFrameDirty()
  }

  private markFrameDirty() {
    this.dirtyFrame = true
    this.requestFrame()
  }

  private markSnapshotDirty() {
    this.snapshotDirty = true
  }

  private flushSnapshot() {
    if (!this.snapshotDirty || !this.snapshotListener) return
    this.snapshotListener(this.getSnapshot())
    this.snapshotDirty = false
  }

  private requestFrame() {
    if (this.disposed || this.animationId !== 0) return
    this.animationId = requestAnimationFrame(this.animate)
  }

  private resizeCanvas() {
    const rect = this.container.getBoundingClientRect()
    this.w = Math.max(1, Math.floor(rect.width))
    this.h = Math.max(1, Math.floor(rect.height))
    this.dpr = Math.min(window.devicePixelRatio || 1, 2)

    this.canvas.width = Math.floor(this.w * this.dpr)
    this.canvas.height = Math.floor(this.h * this.dpr)
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0)
    this.backgroundGradient = null
    this.markFrameDirty()
  }

  private initListeners() {
    this.resizeObserver = new ResizeObserver(() => this.resizeCanvas())
    this.resizeObserver.observe(this.container)
  }
}
