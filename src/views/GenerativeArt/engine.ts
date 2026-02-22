// ── Seeded PRNG (mulberry32) ────────────────────────────────────────
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// ── Simplex-like 2D noise ───────────────────────────────────────────
class SimplexNoise2D {
  private perm: number[]
  private grad: [number, number][] = [
    [1, 1], [-1, 1], [1, -1], [-1, -1],
    [1, 0], [-1, 0], [0, 1], [0, -1]
  ]

  constructor(rng: () => number) {
    const p: number[] = []
    for (let i = 0; i < 256; i++) p[i] = i
    // Fisher-Yates shuffle with seeded rng
    for (let i = 255; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [p[i], p[j]] = [p[j], p[i]]
    }
    this.perm = [...p, ...p]
  }

  private dot(g: [number, number], x: number, y: number): number {
    return g[0] * x + g[1] * y
  }

  noise(xin: number, yin: number): number {
    const F2 = 0.5 * (Math.sqrt(3) - 1)
    const G2 = (3 - Math.sqrt(3)) / 6
    const s = (xin + yin) * F2
    const i = Math.floor(xin + s)
    const j = Math.floor(yin + s)
    const t = (i + j) * G2
    const X0 = i - t
    const Y0 = j - t
    const x0 = xin - X0
    const y0 = yin - Y0

    let i1: number, j1: number
    if (x0 > y0) { i1 = 1; j1 = 0 }
    else { i1 = 0; j1 = 1 }

    const x1 = x0 - i1 + G2
    const y1 = y0 - j1 + G2
    const x2 = x0 - 1 + 2 * G2
    const y2 = y0 - 1 + 2 * G2

    const ii = i & 255
    const jj = j & 255
    const gi0 = this.perm[ii + this.perm[jj]] % 8
    const gi1 = this.perm[ii + i1 + this.perm[jj + j1]] % 8
    const gi2 = this.perm[ii + 1 + this.perm[jj + 1]] % 8

    let n0 = 0, n1 = 0, n2 = 0
    let t0 = 0.5 - x0 * x0 - y0 * y0
    if (t0 >= 0) { t0 *= t0; n0 = t0 * t0 * this.dot(this.grad[gi0], x0, y0) }
    let t1 = 0.5 - x1 * x1 - y1 * y1
    if (t1 >= 0) { t1 *= t1; n1 = t1 * t1 * this.dot(this.grad[gi1], x1, y1) }
    let t2 = 0.5 - x2 * x2 - y2 * y2
    if (t2 >= 0) { t2 *= t2; n2 = t2 * t2 * this.dot(this.grad[gi2], x2, y2) }

    // Result in [-1, 1]
    return 70 * (n0 + n1 + n2)
  }
}

// ── Types ───────────────────────────────────────────────────────────
export type ArtMode = 'flowField' | 'fractalTree' | 'circlePacking' | 'voronoi'

export type PaletteName = 'neonNoir' | 'ocean' | 'sunset' | 'forest' | 'mono'

export interface PaletteInfo {
  name: PaletteName
  label: string
  colors: string[]
}

export const PALETTES: PaletteInfo[] = [
  {
    name: 'neonNoir',
    label: 'Neon Noir',
    colors: ['#ff006e', '#fb5607', '#ffbe0b', '#8338ec', '#3a86ff']
  },
  {
    name: 'ocean',
    label: 'Ocean Depths',
    colors: ['#0077b6', '#00b4d8', '#90e0ef', '#caf0f8', '#03045e']
  },
  {
    name: 'sunset',
    label: 'Sunset',
    colors: ['#ff595e', '#ff924c', '#ffca3a', '#c77dff', '#9b5de5']
  },
  {
    name: 'forest',
    label: 'Forest',
    colors: ['#606c38', '#283618', '#dda15e', '#bc6c25', '#fefae0']
  },
  {
    name: 'mono',
    label: 'Monochrome',
    colors: ['#f8f9fa', '#adb5bd', '#6c757d', '#343a40', '#212529']
  }
]

// ── Helper: parse hex to [r,g,b] 0-255 ─────────────────────────────
function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16)
  return [(v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff]
}

function lerpColor(
  a: [number, number, number],
  b: [number, number, number],
  t: number
): [number, number, number] {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t
  ]
}

// ── Main Engine ─────────────────────────────────────────────────────
export class GenerativeArtEngine {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private animId = 0
  private disposed = false
  private running = false

  private mode: ArtMode = 'flowField'
  private palette: PaletteInfo = PALETTES[0]
  private currentSeed = 0
  private rng!: () => number
  private noise!: SimplexNoise2D

  // Shared animation state
  private startTime = 0
  private frameCount = 0

  // Flow field state
  private ffParticles: { x: number; y: number; vx: number; vy: number; life: number; maxLife: number; colorIdx: number }[] = []
  private ffImageData: ImageData | null = null

  // Fractal tree state
  private ftBranches: { x1: number; y1: number; x2: number; y2: number; depth: number; drawn: boolean; color: string; width: number }[] = []
  private ftLeaves: { x: number; y: number; r: number; color: string; drawn: boolean }[] = []
  private ftDrawIndex = 0
  private ftWindTime = 0
  private ftTreeParams: { baseX: number; baseY: number; angle: number; length: number; decay: number; maxDepth: number; spread: number }[] = []

  // Circle packing state
  private cpCircles: { x: number; y: number; r: number; maxR: number; color: string; growing: boolean; alpha: number }[] = []
  private cpAttempts = 0

  // Voronoi state
  private voPoints: { x: number; y: number; vx: number; vy: number; color: [number, number, number] }[] = []
  private voBuffer: ImageData | null = null
  private voOffCanvas: HTMLCanvasElement | null = null
  private voOffCtx: CanvasRenderingContext2D | null = null

  constructor(private container: HTMLDivElement) {
    this.canvas = document.createElement('canvas')
    this.canvas.style.display = 'block'
    this.canvas.style.width = '100%'
    this.canvas.style.height = '100%'
    container.appendChild(this.canvas)
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })!
    this.resize()
    window.addEventListener('resize', this.handleResize)
  }

  // ── Public API ──────────────────────────────────────────────────
  setMode(mode: ArtMode) {
    this.mode = mode
    this.generate(this.currentSeed)
  }

  setPalette(palette: PaletteInfo) {
    this.palette = palette
    this.generate(this.currentSeed)
  }

  generate(seed?: number) {
    this.stop()
    this.currentSeed = seed ?? Math.floor(Math.random() * 2147483647)
    this.rng = mulberry32(this.currentSeed)
    this.noise = new SimplexNoise2D(this.rng)
    this.frameCount = 0
    this.startTime = performance.now()
    this.running = true
    if (this.disposed) return

    const w = this.canvas.width
    const h = this.canvas.height

    // Clear to background
    this.ctx.fillStyle = '#0a0a0a'
    this.ctx.fillRect(0, 0, w, h)

    switch (this.mode) {
      case 'flowField':
        this.initFlowField()
        break
      case 'fractalTree':
        this.initFractalTree()
        break
      case 'circlePacking':
        this.initCirclePacking()
        break
      case 'voronoi':
        this.initVoronoi()
        break
    }

    this.tick()
    return this.currentSeed
  }

  exportPNG() {
    const link = document.createElement('a')
    link.download = `generative-art-${this.currentSeed}.png`
    link.href = this.canvas.toDataURL('image/png')
    link.click()
  }

  get seed(): number {
    return this.currentSeed
  }

  stop() {
    this.running = false
    if (this.animId) {
      cancelAnimationFrame(this.animId)
      this.animId = 0
    }
  }

  dispose() {
    this.stop()
    this.disposed = true
    window.removeEventListener('resize', this.handleResize)
    if (this.canvas.parentElement) {
      this.canvas.parentElement.removeChild(this.canvas)
    }
  }

  // ── Resize ──────────────────────────────────────────────────────
  private handleResize = () => {
    this.resize()
    if (!this.running) {
      this.generate(this.currentSeed)
    }
  }

  private resize() {
    const dpr = Math.min(window.devicePixelRatio, 2)
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    this.canvas.width = w * dpr
    this.canvas.height = h * dpr
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  // ── Animation Loop ──────────────────────────────────────────────
  private tick = () => {
    if (this.disposed || !this.running) return
    this.animId = requestAnimationFrame(this.tick)
    this.frameCount++

    switch (this.mode) {
      case 'flowField':
        this.drawFlowField()
        break
      case 'fractalTree':
        this.drawFractalTree()
        break
      case 'circlePacking':
        this.drawCirclePacking()
        break
      case 'voronoi':
        this.drawVoronoi()
        break
    }
  }

  // ── Palette helpers ─────────────────────────────────────────────
  private paletteColor(index: number): string {
    const colors = this.palette.colors
    return colors[((index % colors.length) + colors.length) % colors.length]
  }

  private paletteRgb(index: number): [number, number, number] {
    return hexToRgb(this.paletteColor(index))
  }

  private randomPaletteColor(): string {
    return this.paletteColor(Math.floor(this.rng() * this.palette.colors.length))
  }

  // ═══════════════════════════════════════════════════════════════
  // MODE 1: FLOW FIELD
  // ═══════════════════════════════════════════════════════════════
  private initFlowField() {
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    const particleCount = Math.min(3000, Math.floor((w * h) / 200))

    this.ffParticles = []
    for (let i = 0; i < particleCount; i++) {
      this.ffParticles.push({
        x: this.rng() * w,
        y: this.rng() * h,
        vx: 0,
        vy: 0,
        life: 0,
        maxLife: 80 + Math.floor(this.rng() * 120),
        colorIdx: Math.floor(this.rng() * this.palette.colors.length)
      })
    }

    // Use a semi-transparent overlay approach for trails
    this.ffImageData = null
  }

  private drawFlowField() {
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    const ctx = this.ctx
    const zOff = this.frameCount * 0.002

    // Fade existing content slightly to create trails
    ctx.fillStyle = 'rgba(10, 10, 10, 0.012)'
    ctx.fillRect(0, 0, w, h)

    for (const p of this.ffParticles) {
      // Noise angle (use fixed scale, not rng in draw loop)
      const n = this.noise.noise(p.x * 0.003 + zOff, p.y * 0.003)
      const angle = n * Math.PI * 4

      p.vx = Math.cos(angle) * 1.5
      p.vy = Math.sin(angle) * 1.5

      // Draw line segment from old pos to new pos
      const oldX = p.x
      const oldY = p.y
      p.x += p.vx
      p.y += p.vy
      p.life++

      // Determine alpha based on life
      const lifeRatio = p.life / p.maxLife
      const alpha = Math.sin(lifeRatio * Math.PI) * 0.4

      if (alpha > 0.01) {
        const rgb = this.paletteRgb(p.colorIdx)
        // Shift color slightly based on position
        const shift = (p.x / w + p.y / h) * 0.5
        const nextRgb = this.paletteRgb(p.colorIdx + 1)
        const blended = lerpColor(rgb, nextRgb, shift % 1)

        ctx.beginPath()
        ctx.moveTo(oldX, oldY)
        ctx.lineTo(p.x, p.y)
        ctx.strokeStyle = `rgba(${blended[0] | 0},${blended[1] | 0},${blended[2] | 0},${alpha})`
        ctx.lineWidth = 1.2
        ctx.stroke()
      }

      // Respawn if out of bounds or life expired
      if (p.life >= p.maxLife || p.x < 0 || p.x > w || p.y < 0 || p.y > h) {
        p.x = this.rng() * w
        p.y = this.rng() * h
        p.life = 0
        p.maxLife = 80 + Math.floor(this.rng() * 120)
        p.colorIdx = Math.floor(this.rng() * this.palette.colors.length)
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // MODE 2: FRACTAL TREE
  // ═══════════════════════════════════════════════════════════════
  private initFractalTree() {
    const w = this.container.clientWidth
    const h = this.container.clientHeight

    this.ftBranches = []
    this.ftLeaves = []
    this.ftDrawIndex = 0
    this.ftWindTime = 0
    this.ftTreeParams = []

    // Generate 1-3 trees
    const treeCount = 1 + Math.floor(this.rng() * 3)
    for (let t = 0; t < treeCount; t++) {
      const baseX = w * (0.2 + this.rng() * 0.6)
      const baseY = h * 0.9
      const length = h * (0.12 + this.rng() * 0.08)
      const maxDepth = 9 + Math.floor(this.rng() * 3)
      const decay = 0.65 + this.rng() * 0.12
      const spread = 0.3 + this.rng() * 0.4

      this.ftTreeParams.push({ baseX, baseY, angle: -Math.PI / 2, length, decay, maxDepth, spread })
      this.generateBranches(baseX, baseY, -Math.PI / 2, length, maxDepth, decay, spread, 0)
    }
  }

  private generateBranches(
    x: number, y: number, angle: number,
    length: number, maxDepth: number,
    decay: number, spread: number, depth: number
  ) {
    if (depth >= maxDepth || length < 2) {
      // Add leaf
      this.ftLeaves.push({
        x, y,
        r: 2 + this.rng() * 4,
        color: this.randomPaletteColor(),
        drawn: false
      })
      return
    }

    const endX = x + Math.cos(angle) * length
    const endY = y + Math.sin(angle) * length

    const depthRatio = depth / maxDepth
    const colorIdx = Math.floor(depthRatio * (this.palette.colors.length - 1))
    const color = this.paletteColor(colorIdx)
    const lineWidth = Math.max(1, (1 - depthRatio) * 6)

    this.ftBranches.push({
      x1: x, y1: y, x2: endX, y2: endY,
      depth, drawn: false, color, width: lineWidth
    })

    // Branch into 2-3 sub-branches
    const numBranches = 2 + (this.rng() > 0.7 ? 1 : 0)
    for (let i = 0; i < numBranches; i++) {
      const angleOffset = (i - (numBranches - 1) / 2) * spread + (this.rng() - 0.5) * 0.2
      const newLength = length * (decay + (this.rng() - 0.5) * 0.1)
      this.generateBranches(endX, endY, angle + angleOffset, newLength, maxDepth, decay, spread, depth + 1)
    }
  }

  private drawFractalTree() {
    const ctx = this.ctx
    const batchSize = Math.max(8, Math.floor(this.ftBranches.length / 90))

    // Draw branches progressively
    let drawn = 0
    for (let i = this.ftDrawIndex; i < this.ftBranches.length && drawn < batchSize; i++) {
      const b = this.ftBranches[i]
      if (b.drawn) continue

      // Apply wind sway
      this.ftWindTime += 0.00005
      const sway = Math.sin(this.ftWindTime * 2 + b.depth * 0.3) * b.depth * 0.4

      ctx.beginPath()
      ctx.moveTo(b.x1 + sway * 0.3, b.y1)
      ctx.lineTo(b.x2 + sway, b.y2)
      ctx.strokeStyle = b.color
      ctx.lineWidth = b.width
      ctx.lineCap = 'round'
      ctx.globalAlpha = 0.85
      ctx.stroke()

      // Glow effect
      ctx.shadowColor = b.color
      ctx.shadowBlur = 4
      ctx.stroke()
      ctx.shadowBlur = 0

      ctx.globalAlpha = 1
      b.drawn = true
      drawn++
      this.ftDrawIndex = i + 1
    }

    // After all branches, draw leaves
    if (this.ftDrawIndex >= this.ftBranches.length) {
      let leafDrawn = 0
      const leafBatch = Math.max(4, Math.floor(this.ftLeaves.length / 60))
      for (const leaf of this.ftLeaves) {
        if (leaf.drawn || leafDrawn >= leafBatch) continue
        const sway = Math.sin(this.ftWindTime * 3 + leaf.x * 0.01) * 2

        ctx.beginPath()
        ctx.arc(leaf.x + sway, leaf.y, leaf.r, 0, Math.PI * 2)
        ctx.fillStyle = leaf.color
        ctx.globalAlpha = 0.7
        ctx.fill()
        ctx.globalAlpha = 1

        leaf.drawn = true
        leafDrawn++
      }

      // Check if all done
      if (this.ftLeaves.every(l => l.drawn)) {
        // Keep running for wind animation - redraw everything with wind
        this.animateTreeWind()
      }
    }
  }

  private animateTreeWind() {
    const ctx = this.ctx
    const w = this.container.clientWidth
    const h = this.container.clientHeight

    this.ftWindTime += 0.02

    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, w, h)

    // Redraw all branches with wind
    for (const b of this.ftBranches) {
      const sway = Math.sin(this.ftWindTime + b.depth * 0.4) * b.depth * 0.5

      ctx.beginPath()
      ctx.moveTo(b.x1 + sway * 0.3, b.y1)
      ctx.lineTo(b.x2 + sway, b.y2)
      ctx.strokeStyle = b.color
      ctx.lineWidth = b.width
      ctx.lineCap = 'round'
      ctx.globalAlpha = 0.85
      ctx.stroke()
      ctx.globalAlpha = 1
    }

    // Redraw leaves with wind
    for (const leaf of this.ftLeaves) {
      const sway = Math.sin(this.ftWindTime * 1.5 + leaf.x * 0.01) * 3

      ctx.beginPath()
      ctx.arc(leaf.x + sway, leaf.y, leaf.r, 0, Math.PI * 2)
      ctx.fillStyle = leaf.color
      ctx.globalAlpha = 0.7
      ctx.fill()
      ctx.globalAlpha = 1
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // MODE 3: CIRCLE PACKING
  // ═══════════════════════════════════════════════════════════════
  private initCirclePacking() {
    this.cpCircles = []
    this.cpAttempts = 0
  }

  private drawCirclePacking() {
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    const ctx = this.ctx
    const maxAttempts = 500

    // Try to place new circles each frame
    for (let attempt = 0; attempt < 12; attempt++) {
      if (this.cpAttempts > maxAttempts * 30) {
        // Done - just keep rendering existing circles with subtle animation
        this.animateCircles()
        return
      }
      this.cpAttempts++

      const x = this.rng() * w
      const y = this.rng() * h
      const minR = 2
      const maxR = 20 + this.rng() * 80

      // Check if position is valid (not overlapping existing circles)
      let valid = true
      let maxPossibleR = maxR
      for (const c of this.cpCircles) {
        const dx = x - c.x
        const dy = y - c.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < c.r + minR + 2) {
          valid = false
          break
        }
        maxPossibleR = Math.min(maxPossibleR, dist - c.r - 2)
      }

      // Check edges
      maxPossibleR = Math.min(maxPossibleR, x, y, w - x, h - y)

      if (valid && maxPossibleR >= minR) {
        const colorIdx = Math.floor(this.rng() * this.palette.colors.length)
        this.cpCircles.push({
          x, y,
          r: minR,
          maxR: Math.min(maxPossibleR, maxR),
          color: this.paletteColor(colorIdx),
          growing: true,
          alpha: 0
        })
      }
    }

    // Grow existing circles
    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, w, h)

    for (const c of this.cpCircles) {
      if (c.growing) {
        c.r += 0.5
        c.alpha = Math.min(c.alpha + 0.05, 0.85)

        // Check if we should stop growing
        if (c.r >= c.maxR) {
          c.growing = false
          c.r = c.maxR
        }

        // Check collision with other circles
        for (const other of this.cpCircles) {
          if (other === c) continue
          const dx = c.x - other.x
          const dy = c.y - other.y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < c.r + other.r + 2) {
            c.growing = false
            c.r = Math.max(c.r - 0.5, 2)
            break
          }
        }
      }

      // Draw circle
      const rgb = hexToRgb(c.color)

      // Filled circle with gradient
      const grad = ctx.createRadialGradient(c.x - c.r * 0.3, c.y - c.r * 0.3, 0, c.x, c.y, c.r)
      grad.addColorStop(0, `rgba(${Math.min(rgb[0] + 60, 255)},${Math.min(rgb[1] + 60, 255)},${Math.min(rgb[2] + 60, 255)},${c.alpha})`)
      grad.addColorStop(0.7, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${c.alpha * 0.8})`)
      grad.addColorStop(1, `rgba(${rgb[0] >> 1},${rgb[1] >> 1},${rgb[2] >> 1},${c.alpha * 0.6})`)

      ctx.beginPath()
      ctx.arc(c.x, c.y, c.r, 0, Math.PI * 2)
      ctx.fillStyle = grad
      ctx.fill()

      // Subtle stroke
      ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${c.alpha * 0.3})`
      ctx.lineWidth = 1
      ctx.stroke()
    }
  }

  private animateCircles() {
    const ctx = this.ctx
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    const time = this.frameCount * 0.02

    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, w, h)

    for (const c of this.cpCircles) {
      const rgb = hexToRgb(c.color)
      const pulse = 1 + Math.sin(time + c.x * 0.01 + c.y * 0.01) * 0.02

      const grad = ctx.createRadialGradient(
        c.x - c.r * 0.3, c.y - c.r * 0.3, 0,
        c.x, c.y, c.r * pulse
      )
      grad.addColorStop(0, `rgba(${Math.min(rgb[0] + 60, 255)},${Math.min(rgb[1] + 60, 255)},${Math.min(rgb[2] + 60, 255)},${c.alpha})`)
      grad.addColorStop(0.7, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${c.alpha * 0.8})`)
      grad.addColorStop(1, `rgba(${rgb[0] >> 1},${rgb[1] >> 1},${rgb[2] >> 1},${c.alpha * 0.6})`)

      ctx.beginPath()
      ctx.arc(c.x, c.y, c.r * pulse, 0, Math.PI * 2)
      ctx.fillStyle = grad
      ctx.fill()

      ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${c.alpha * 0.3})`
      ctx.lineWidth = 1
      ctx.stroke()
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // MODE 4: VORONOI CRYSTAL
  // ═══════════════════════════════════════════════════════════════
  private initVoronoi() {
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    const count = 40 + Math.floor(this.rng() * 40)

    this.voPoints = []
    for (let i = 0; i < count; i++) {
      const colorIdx = Math.floor(this.rng() * this.palette.colors.length)
      const rgb = this.paletteRgb(colorIdx)
      this.voPoints.push({
        x: this.rng() * w,
        y: this.rng() * h,
        vx: (this.rng() - 0.5) * 0.5,
        vy: (this.rng() - 0.5) * 0.5,
        color: rgb
      })
    }
    this.voBuffer = null
  }

  private drawVoronoi() {
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    const ctx = this.ctx
    const dpr = Math.min(window.devicePixelRatio, 2)

    // Move points slowly
    for (const p of this.voPoints) {
      p.x += p.vx
      p.y += p.vy

      // Bounce off edges
      if (p.x < 0 || p.x > w) p.vx *= -1
      if (p.y < 0 || p.y > h) p.vy *= -1
      p.x = Math.max(0, Math.min(w, p.x))
      p.y = Math.max(0, Math.min(h, p.y))
    }

    // Render Voronoi via per-pixel nearest-point calculation
    // Use lower resolution for performance
    const scale = 4
    const sw = Math.ceil(w / scale)
    const sh = Math.ceil(h / scale)

    // Reuse offscreen canvas for low-res render
    if (!this.voOffCanvas || this.voOffCanvas.width !== sw || this.voOffCanvas.height !== sh) {
      this.voOffCanvas = document.createElement('canvas')
      this.voOffCanvas.width = sw
      this.voOffCanvas.height = sh
      this.voOffCtx = this.voOffCanvas.getContext('2d')!
    }
    const offCtx = this.voOffCtx!
    const offCanvas = this.voOffCanvas
    const imageData = offCtx.createImageData(sw, sh)
    const data = imageData.data

    for (let py = 0; py < sh; py++) {
      for (let px = 0; px < sw; px++) {
        const rx = px * scale
        const ry = py * scale

        // Find closest and second-closest point
        let minDist = Infinity
        let secondDist = Infinity
        let closestIdx = 0

        for (let i = 0; i < this.voPoints.length; i++) {
          const dx = rx - this.voPoints[i].x
          const dy = ry - this.voPoints[i].y
          const dist = dx * dx + dy * dy
          if (dist < minDist) {
            secondDist = minDist
            minDist = dist
            closestIdx = i
          } else if (dist < secondDist) {
            secondDist = dist
          }
        }

        const closest = this.voPoints[closestIdx]
        const d1 = Math.sqrt(minDist)
        const d2 = Math.sqrt(secondDist)
        const edgeFactor = 1 - Math.pow(Math.max(0, 1 - (d2 - d1) / 12), 2)

        // Iridescent color shift based on distance
        const distNorm = d1 / 80
        const shiftR = Math.sin(distNorm * Math.PI * 2) * 30
        const shiftG = Math.sin(distNorm * Math.PI * 2 + 2.094) * 30
        const shiftB = Math.sin(distNorm * Math.PI * 2 + 4.189) * 30

        const baseColor = closest.color
        const idx = (py * sw + px) * 4

        // Darken near edges for crystal facet effect
        const brightness = 0.5 + edgeFactor * 0.5
        const edgeDark = d2 - d1 < 3 ? 0.15 : 1

        data[idx] = Math.max(0, Math.min(255, (baseColor[0] + shiftR) * brightness * edgeDark))
        data[idx + 1] = Math.max(0, Math.min(255, (baseColor[1] + shiftG) * brightness * edgeDark))
        data[idx + 2] = Math.max(0, Math.min(255, (baseColor[2] + shiftB) * brightness * edgeDark))
        data[idx + 3] = 255
      }
    }

    offCtx.putImageData(imageData, 0, 0)

    // Draw scaled up with smoothing
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(offCanvas, 0, 0, w, h)

    // Draw cell edges (Voronoi edges) as bright lines
    ctx.globalAlpha = 0.15
    for (let i = 0; i < this.voPoints.length; i++) {
      const p = this.voPoints[i]
      ctx.beginPath()
      ctx.arc(p.x, p.y, 2, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(255,255,255,0.3)`
      ctx.fill()
    }
    ctx.globalAlpha = 1
  }
}
