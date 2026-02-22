/**
 * Boids flocking simulation using Craig Reynolds' algorithm.
 *
 * Features:
 *   - 1000-3000 boids as directional triangles
 *   - Three classic rules: separation, alignment, cohesion
 *   - Spatial hashing for efficient O(n) neighbor lookup
 *   - Mouse interaction (attract / repel / none)
 *   - Optional predator that chases nearest boid
 *   - Optional trails (last 5 positions)
 *   - HSL heading-based coloring
 *   - Toroidal (wrapping) space
 */

// ---------------------------------------------------------------------------
// Vec2 helpers
// ---------------------------------------------------------------------------

interface Vec2 {
  x: number
  y: number
}

function v2(x: number, y: number): Vec2 {
  return { x, y }
}

function v2add(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, y: a.y + b.y }
}

function v2sub(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, y: a.y - b.y }
}

function v2scale(a: Vec2, s: number): Vec2 {
  return { x: a.x * s, y: a.y * s }
}

function v2len(a: Vec2): number {
  return Math.sqrt(a.x * a.x + a.y * a.y)
}

function v2lenSq(a: Vec2): number {
  return a.x * a.x + a.y * a.y
}

function v2norm(a: Vec2): Vec2 {
  const l = v2len(a)
  if (l < 1e-10) return { x: 0, y: 0 }
  return { x: a.x / l, y: a.y / l }
}

function v2setMag(a: Vec2, mag: number): Vec2 {
  const l = v2len(a)
  if (l < 1e-10) return { x: 0, y: 0 }
  const s = mag / l
  return { x: a.x * s, y: a.y * s }
}

function v2limit(a: Vec2, max: number): Vec2 {
  const lsq = v2lenSq(a)
  if (lsq > max * max) {
    return v2setMag(a, max)
  }
  return a
}

// ---------------------------------------------------------------------------
// Boid
// ---------------------------------------------------------------------------

interface Boid {
  pos: Vec2
  vel: Vec2
  acc: Vec2
  trail: Vec2[]
}

// ---------------------------------------------------------------------------
// Predator
// ---------------------------------------------------------------------------

interface Predator {
  pos: Vec2
  vel: Vec2
  acc: Vec2
}

// ---------------------------------------------------------------------------
// Spatial hash grid
// ---------------------------------------------------------------------------

class SpatialHash {
  private cellSize: number
  private cols: number
  private rows: number
  private grid: Int32Array
  private counts: Int32Array
  private cellCapacity: number

  constructor(width: number, height: number, cellSize: number, maxBoidsPerCell: number) {
    this.cellSize = cellSize
    this.cols = Math.ceil(width / cellSize)
    this.rows = Math.ceil(height / cellSize)
    this.cellCapacity = maxBoidsPerCell
    const totalCells = this.cols * this.rows
    this.grid = new Int32Array(totalCells * maxBoidsPerCell)
    this.counts = new Int32Array(totalCells)
  }

  clear() {
    this.counts.fill(0)
  }

  insert(index: number, x: number, y: number) {
    const col = Math.floor(x / this.cellSize)
    const row = Math.floor(y / this.cellSize)
    if (col < 0 || col >= this.cols || row < 0 || row >= this.rows) return
    const cellIdx = row * this.cols + col
    const count = this.counts[cellIdx]
    if (count < this.cellCapacity) {
      this.grid[cellIdx * this.cellCapacity + count] = index
      this.counts[cellIdx] = count + 1
    }
  }

  query(x: number, y: number, radius: number, callback: (index: number) => void) {
    const minCol = Math.max(0, Math.floor((x - radius) / this.cellSize))
    const maxCol = Math.min(this.cols - 1, Math.floor((x + radius) / this.cellSize))
    const minRow = Math.max(0, Math.floor((y - radius) / this.cellSize))
    const maxRow = Math.min(this.rows - 1, Math.floor((y + radius) / this.cellSize))

    for (let r = minRow; r <= maxRow; r++) {
      for (let c = minCol; c <= maxCol; c++) {
        const cellIdx = r * this.cols + c
        const count = this.counts[cellIdx]
        const base = cellIdx * this.cellCapacity
        for (let i = 0; i < count; i++) {
          callback(this.grid[base + i])
        }
      }
    }
  }

  resize(width: number, height: number) {
    this.cols = Math.ceil(width / this.cellSize)
    this.rows = Math.ceil(height / this.cellSize)
    const totalCells = this.cols * this.rows
    this.grid = new Int32Array(totalCells * this.cellCapacity)
    this.counts = new Int32Array(totalCells)
  }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

type MouseMode = 'attract' | 'repel' | 'none'

export class BoidsEngine {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D

  private boids: Boid[] = []
  private predator: Predator | null = null
  private hasPredator = false
  private showTrails = false

  private separationWeight = 1.5
  private alignmentWeight = 1.0
  private cohesionWeight = 1.0
  private perceptionRadius = 50

  private maxSpeed = 3
  private minSpeed = 1.5
  private maxForce = 0.15

  private mouseMode: MouseMode = 'none'
  private mousePos: Vec2 = v2(0, 0)
  private mouseInCanvas = false
  private mouseInfluenceRadius = 150

  private spatialHash!: SpatialHash
  private w = 0
  private h = 0

  private animationId = 0
  private disposed = false
  private resizeObserver: ResizeObserver | null = null

  constructor(private container: HTMLDivElement) {
    this.canvas = document.createElement('canvas')
    this.canvas.style.width = '100%'
    this.canvas.style.height = '100%'
    this.canvas.style.display = 'block'
    container.appendChild(this.canvas)

    this.ctx = this.canvas.getContext('2d')!

    this.resizeCanvas()
    this.spatialHash = new SpatialHash(this.w, this.h, this.perceptionRadius, 30)
    this.initBoids(1500)
    this.initListeners()
    this.animate()
  }

  // --- Public API ---

  setBoidCount(n: number) {
    const target = Math.max(100, Math.min(n, 5000))
    if (target > this.boids.length) {
      while (this.boids.length < target) {
        this.boids.push(this.createRandomBoid())
      }
    } else {
      this.boids.length = target
    }
  }

  setSeparation(v: number) { this.separationWeight = v }
  setAlignment(v: number) { this.alignmentWeight = v }
  setCohesion(v: number) { this.cohesionWeight = v }

  setPerceptionRadius(v: number) {
    this.perceptionRadius = v
    this.spatialHash = new SpatialHash(this.w, this.h, v, 30)
  }

  setMouseMode(mode: MouseMode) { this.mouseMode = mode }

  togglePredator(): boolean {
    this.hasPredator = !this.hasPredator
    if (this.hasPredator && !this.predator) {
      this.predator = {
        pos: v2(this.w * 0.5, this.h * 0.5),
        vel: v2(Math.random() * 2 - 1, Math.random() * 2 - 1),
        acc: v2(0, 0),
      }
    }
    if (!this.hasPredator) this.predator = null
    return this.hasPredator
  }

  toggleTrails(): boolean {
    this.showTrails = !this.showTrails
    if (!this.showTrails) {
      for (const b of this.boids) b.trail = []
    }
    return this.showTrails
  }

  get boidCount(): number {
    return this.boids.length
  }

  dispose() {
    this.disposed = true
    cancelAnimationFrame(this.animationId)
    this.removeListeners()
    if (this.resizeObserver) {
      this.resizeObserver.disconnect()
      this.resizeObserver = null
    }
    if (this.canvas.parentElement) {
      this.canvas.parentElement.removeChild(this.canvas)
    }
  }

  // --- Init ---

  private resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio, 2)
    this.w = this.container.clientWidth
    this.h = this.container.clientHeight
    this.canvas.width = this.w * dpr
    this.canvas.height = this.h * dpr
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  private createRandomBoid(): Boid {
    const angle = Math.random() * Math.PI * 2
    const speed = this.minSpeed + Math.random() * (this.maxSpeed - this.minSpeed)
    return {
      pos: v2(Math.random() * this.w, Math.random() * this.h),
      vel: v2(Math.cos(angle) * speed, Math.sin(angle) * speed),
      acc: v2(0, 0),
      trail: [],
    }
  }

  private initBoids(n: number) {
    this.boids = []
    for (let i = 0; i < n; i++) {
      this.boids.push(this.createRandomBoid())
    }
  }

  // --- Input ---

  private onMouseMove = (e: MouseEvent) => {
    const rect = this.canvas.getBoundingClientRect()
    this.mousePos = v2(e.clientX - rect.left, e.clientY - rect.top)
    this.mouseInCanvas = true
  }

  private onMouseLeave = () => {
    this.mouseInCanvas = false
  }

  private onTouchMove = (e: TouchEvent) => {
    e.preventDefault()
    const rect = this.canvas.getBoundingClientRect()
    const touch = e.touches[0]
    this.mousePos = v2(touch.clientX - rect.left, touch.clientY - rect.top)
    this.mouseInCanvas = true
  }

  private onTouchEnd = () => {
    this.mouseInCanvas = false
  }

  private initListeners() {
    this.canvas.addEventListener('mousemove', this.onMouseMove)
    this.canvas.addEventListener('mouseleave', this.onMouseLeave)
    this.canvas.addEventListener('touchmove', this.onTouchMove, { passive: false })
    window.addEventListener('touchend', this.onTouchEnd)

    this.resizeObserver = new ResizeObserver(() => {
      if (!this.disposed) {
        this.resizeCanvas()
        this.spatialHash.resize(this.w, this.h)
      }
    })
    this.resizeObserver.observe(this.container)
  }

  private removeListeners() {
    this.canvas.removeEventListener('mousemove', this.onMouseMove)
    this.canvas.removeEventListener('mouseleave', this.onMouseLeave)
    this.canvas.removeEventListener('touchmove', this.onTouchMove)
    window.removeEventListener('touchend', this.onTouchEnd)
  }

  // --- Flocking ---

  private wrapDelta(dx: number, dy: number): Vec2 {
    if (dx > this.w * 0.5) dx -= this.w
    else if (dx < -this.w * 0.5) dx += this.w
    if (dy > this.h * 0.5) dy -= this.h
    else if (dy < -this.h * 0.5) dy += this.h
    return v2(dx, dy)
  }

  private updateBoids() {
    const w = this.w
    const h = this.h
    const boids = this.boids
    const n = boids.length
    const pr = this.perceptionRadius
    const prSq = pr * pr

    // Rebuild spatial hash
    this.spatialHash.clear()
    for (let i = 0; i < n; i++) {
      // Wrap position for insertion
      const b = boids[i]
      this.spatialHash.insert(i, b.pos.x, b.pos.y)

      // Also insert ghost copies near edges for wrapping queries
      if (b.pos.x < pr) this.spatialHash.insert(i, b.pos.x + w, b.pos.y)
      else if (b.pos.x > w - pr) this.spatialHash.insert(i, b.pos.x - w, b.pos.y)
      if (b.pos.y < pr) this.spatialHash.insert(i, b.pos.x, b.pos.y + h)
      else if (b.pos.y > h - pr) this.spatialHash.insert(i, b.pos.x, b.pos.y - h)
    }

    // Compute steering forces for each boid
    for (let i = 0; i < n; i++) {
      const boid = boids[i]
      let sepX = 0, sepY = 0, sepCount = 0
      let aliX = 0, aliY = 0, aliCount = 0
      let cohX = 0, cohY = 0, cohCount = 0

      this.spatialHash.query(boid.pos.x, boid.pos.y, pr, (j: number) => {
        if (j === i) return
        const other = boids[j]
        const delta = this.wrapDelta(other.pos.x - boid.pos.x, other.pos.y - boid.pos.y)
        const dSq = v2lenSq(delta)
        if (dSq > prSq || dSq < 1e-6) return

        const d = Math.sqrt(dSq)

        // Separation - weight inversely by distance
        sepX -= delta.x / d
        sepY -= delta.y / d
        sepCount++

        // Alignment
        aliX += other.vel.x
        aliY += other.vel.y
        aliCount++

        // Cohesion
        cohX += delta.x
        cohY += delta.y
        cohCount++
      })

      let steerX = 0, steerY = 0

      // Separation
      if (sepCount > 0) {
        sepX /= sepCount
        sepY /= sepCount
        const sepMag = v2len(v2(sepX, sepY))
        if (sepMag > 0) {
          sepX = sepX / sepMag * this.maxSpeed - boid.vel.x
          sepY = sepY / sepMag * this.maxSpeed - boid.vel.y
          const l = v2len(v2(sepX, sepY))
          if (l > this.maxForce) {
            sepX = sepX / l * this.maxForce
            sepY = sepY / l * this.maxForce
          }
        }
        steerX += sepX * this.separationWeight
        steerY += sepY * this.separationWeight
      }

      // Alignment
      if (aliCount > 0) {
        aliX /= aliCount
        aliY /= aliCount
        const aliMag = v2len(v2(aliX, aliY))
        if (aliMag > 0) {
          aliX = aliX / aliMag * this.maxSpeed - boid.vel.x
          aliY = aliY / aliMag * this.maxSpeed - boid.vel.y
          const l = v2len(v2(aliX, aliY))
          if (l > this.maxForce) {
            aliX = aliX / l * this.maxForce
            aliY = aliY / l * this.maxForce
          }
        }
        steerX += aliX * this.alignmentWeight
        steerY += aliY * this.alignmentWeight
      }

      // Cohesion
      if (cohCount > 0) {
        cohX /= cohCount
        cohY /= cohCount
        const cohMag = v2len(v2(cohX, cohY))
        if (cohMag > 0) {
          cohX = cohX / cohMag * this.maxSpeed - boid.vel.x
          cohY = cohY / cohMag * this.maxSpeed - boid.vel.y
          const l = v2len(v2(cohX, cohY))
          if (l > this.maxForce) {
            cohX = cohX / l * this.maxForce
            cohY = cohY / l * this.maxForce
          }
        }
        steerX += cohX * this.cohesionWeight
        steerY += cohY * this.cohesionWeight
      }

      // Mouse influence
      if (this.mouseInCanvas && this.mouseMode !== 'none') {
        const mdx = this.mousePos.x - boid.pos.x
        const mdy = this.mousePos.y - boid.pos.y
        const mdSq = mdx * mdx + mdy * mdy
        if (mdSq < this.mouseInfluenceRadius * this.mouseInfluenceRadius && mdSq > 1) {
          const md = Math.sqrt(mdSq)
          const factor = this.mouseMode === 'attract' ? 1 : -1
          const mx = (mdx / md * this.maxSpeed - boid.vel.x) * factor
          const my = (mdy / md * this.maxSpeed - boid.vel.y) * factor
          const ml = v2len(v2(mx, my))
          const cap = this.maxForce * 3
          if (ml > cap) {
            steerX += mx / ml * cap
            steerY += my / ml * cap
          } else {
            steerX += mx
            steerY += my
          }
        }
      }

      // Predator avoidance
      if (this.predator) {
        const pd = this.wrapDelta(this.predator.pos.x - boid.pos.x, this.predator.pos.y - boid.pos.y)
        const pdSq = v2lenSq(pd)
        const scatterRadius = pr * 2
        if (pdSq < scatterRadius * scatterRadius && pdSq > 1) {
          const pdd = Math.sqrt(pdSq)
          const fleeX = (-pd.x / pdd * this.maxSpeed - boid.vel.x)
          const fleeY = (-pd.y / pdd * this.maxSpeed - boid.vel.y)
          const fl = v2len(v2(fleeX, fleeY))
          const cap = this.maxForce * 4
          if (fl > cap) {
            steerX += fleeX / fl * cap
            steerY += fleeY / fl * cap
          } else {
            steerX += fleeX
            steerY += fleeY
          }
        }
      }

      boid.acc = v2(steerX, steerY)
    }

    // Integrate
    for (let i = 0; i < n; i++) {
      const b = boids[i]

      // Store trail
      if (this.showTrails) {
        b.trail.push(v2(b.pos.x, b.pos.y))
        if (b.trail.length > 5) b.trail.shift()
      }

      b.vel = v2add(b.vel, b.acc)

      // Clamp speed
      const speed = v2len(b.vel)
      if (speed > this.maxSpeed) {
        b.vel = v2setMag(b.vel, this.maxSpeed)
      } else if (speed < this.minSpeed) {
        b.vel = v2setMag(b.vel, this.minSpeed)
      }

      b.pos = v2add(b.pos, b.vel)
      b.acc = v2(0, 0)

      // Wrap around edges
      if (b.pos.x < 0) b.pos.x += w
      else if (b.pos.x > w) b.pos.x -= w
      if (b.pos.y < 0) b.pos.y += h
      else if (b.pos.y > h) b.pos.y -= h
    }

    // Update predator
    if (this.predator) {
      this.updatePredator()
    }
  }

  private updatePredator() {
    const pred = this.predator!
    const w = this.w
    const h = this.h

    // Chase nearest boid
    let nearestDist = Infinity
    let nearestBoid: Boid | null = null
    for (const b of this.boids) {
      const d = this.wrapDelta(b.pos.x - pred.pos.x, b.pos.y - pred.pos.y)
      const dSq = v2lenSq(d)
      if (dSq < nearestDist) {
        nearestDist = dSq
        nearestBoid = b
      }
    }

    if (nearestBoid) {
      const d = this.wrapDelta(nearestBoid.pos.x - pred.pos.x, nearestBoid.pos.y - pred.pos.y)
      const dn = v2norm(d)
      const desired = v2scale(dn, this.maxSpeed * 0.9)
      const steer = v2limit(v2sub(desired, pred.vel), this.maxForce * 2)
      pred.acc = steer
    }

    pred.vel = v2add(pred.vel, pred.acc)
    const pSpeed = v2len(pred.vel)
    if (pSpeed > this.maxSpeed * 0.9) {
      pred.vel = v2setMag(pred.vel, this.maxSpeed * 0.9)
    }
    pred.pos = v2add(pred.pos, pred.vel)
    pred.acc = v2(0, 0)

    if (pred.pos.x < 0) pred.pos.x += w
    else if (pred.pos.x > w) pred.pos.x -= w
    if (pred.pos.y < 0) pred.pos.y += h
    else if (pred.pos.y > h) pred.pos.y -= h
  }

  // --- Rendering ---

  private draw() {
    const ctx = this.ctx
    const w = this.w
    const h = this.h
    const boids = this.boids

    // Clear
    ctx.fillStyle = '#060612'
    ctx.fillRect(0, 0, w, h)

    // Draw trails
    if (this.showTrails) {
      for (const b of boids) {
        if (b.trail.length < 2) continue
        const heading = Math.atan2(b.vel.y, b.vel.x)
        const hue = ((heading / (Math.PI * 2)) * 360 + 360) % 360
        ctx.beginPath()
        ctx.moveTo(b.trail[0].x, b.trail[0].y)
        for (let t = 1; t < b.trail.length; t++) {
          ctx.lineTo(b.trail[t].x, b.trail[t].y)
        }
        ctx.lineTo(b.pos.x, b.pos.y)
        ctx.strokeStyle = `hsla(${hue | 0}, 80%, 60%, 0.2)`
        ctx.lineWidth = 1
        ctx.stroke()
      }
    }

    // Draw boids as triangles
    const size = 4
    for (const b of boids) {
      const heading = Math.atan2(b.vel.y, b.vel.x)
      const hue = ((heading / (Math.PI * 2)) * 360 + 360) % 360

      ctx.save()
      ctx.translate(b.pos.x, b.pos.y)
      ctx.rotate(heading)

      ctx.fillStyle = `hsl(${hue | 0}, 80%, 60%)`
      ctx.beginPath()
      ctx.moveTo(size * 2, 0)
      ctx.lineTo(-size, size)
      ctx.lineTo(-size, -size)
      ctx.closePath()
      ctx.fill()

      ctx.restore()
    }

    // Draw predator
    if (this.predator) {
      const p = this.predator
      const heading = Math.atan2(p.vel.y, p.vel.x)

      ctx.save()
      ctx.translate(p.pos.x, p.pos.y)
      ctx.rotate(heading)

      ctx.fillStyle = '#ff3030'
      ctx.beginPath()
      ctx.moveTo(14, 0)
      ctx.lineTo(-7, 7)
      ctx.lineTo(-4, 0)
      ctx.lineTo(-7, -7)
      ctx.closePath()
      ctx.fill()

      ctx.shadowColor = '#ff3030'
      ctx.shadowBlur = 12
      ctx.fill()
      ctx.shadowBlur = 0

      ctx.restore()
    }

    // Mouse indicator
    if (this.mouseInCanvas && this.mouseMode !== 'none') {
      ctx.strokeStyle = this.mouseMode === 'attract'
        ? 'rgba(100, 200, 255, 0.2)'
        : 'rgba(255, 100, 100, 0.2)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.arc(this.mousePos.x, this.mousePos.y, this.mouseInfluenceRadius, 0, Math.PI * 2)
      ctx.stroke()
    }
  }

  // --- Main loop ---

  private animate = () => {
    if (this.disposed) return
    this.animationId = requestAnimationFrame(this.animate)
    this.updateBoids()
    this.draw()
  }
}
