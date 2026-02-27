/**
 * 2D rigid body physics engine with impulse-based collision response.
 *
 * Features:
 *   - Circle and box rigid bodies
 *   - Circle-circle, circle-box, and box-box (SAT) collision detection
 *   - Impulse-based collision response with restitution
 *   - Configurable gravity
 *   - Distance constraints (springs) between bodies
 *   - Boundary walls at canvas edges
 *   - Tools: circle, box, spring, gravity gun, explosion
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Tool = 'circle' | 'box' | 'spring' | 'gravity' | 'explosion'

interface Vec2 {
  x: number
  y: number
}

function vec2(x: number, y: number): Vec2 {
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

function v2dot(a: Vec2, b: Vec2): number {
  return a.x * b.x + a.y * b.y
}

function v2cross(a: Vec2, b: Vec2): number {
  return a.x * b.y - a.y * b.x
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

function v2perp(a: Vec2): Vec2 {
  return { x: -a.y, y: a.x }
}

function v2neg(a: Vec2): Vec2 {
  return { x: -a.x, y: -a.y }
}

function v2rotate(v: Vec2, angle: number): Vec2 {
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  return { x: v.x * c - v.y * s, y: v.x * s + v.y * c }
}

function clamp(val: number, min: number, max: number): number {
  return val < min ? min : val > max ? max : val
}

// ---------------------------------------------------------------------------
// Body
// ---------------------------------------------------------------------------

type BodyType = 'circle' | 'box'

interface Body {
  type: BodyType
  pos: Vec2
  vel: Vec2
  angle: number
  angVel: number
  mass: number
  invMass: number
  inertia: number
  invInertia: number
  restitution: number
  // Shape data
  radius: number     // circle radius, or half-diagonal for boxes (used for broad phase)
  halfW: number      // box half-width
  halfH: number      // box half-height
  // Rendering
  color: string
  id: number
  isStatic: boolean
}

let bodyIdCounter = 0

function createCircle(x: number, y: number, radius: number, restitution: number): Body {
  const r = Math.max(radius, 5)
  const mass = Math.PI * r * r * 0.01
  const inertia = 0.5 * mass * r * r
  return {
    type: 'circle',
    pos: vec2(x, y),
    vel: vec2(0, 0),
    angle: 0,
    angVel: 0,
    mass,
    invMass: 1 / mass,
    inertia,
    invInertia: 1 / inertia,
    restitution,
    radius: r,
    halfW: 0,
    halfH: 0,
    color: randomPastelColor(),
    id: bodyIdCounter++,
    isStatic: false,
  }
}

function createBox(x: number, y: number, halfW: number, halfH: number, restitution: number): Body {
  const hw = Math.max(halfW, 5)
  const hh = Math.max(halfH, 5)
  const mass = hw * hh * 4 * 0.01
  const inertia = (mass / 12) * (4 * hw * hw + 4 * hh * hh)
  return {
    type: 'box',
    pos: vec2(x, y),
    vel: vec2(0, 0),
    angle: 0,
    angVel: 0,
    mass,
    invMass: 1 / mass,
    inertia,
    invInertia: 1 / inertia,
    restitution,
    radius: Math.sqrt(hw * hw + hh * hh),
    halfW: hw,
    halfH: hh,
    color: randomPastelColor(),
    id: bodyIdCounter++,
    isStatic: false,
  }
}

function randomPastelColor(): string {
  const h = Math.random() * 360
  const s = 50 + Math.random() * 30
  const l = 55 + Math.random() * 20
  return `hsl(${h | 0}, ${s | 0}%, ${l | 0}%)`
}

// ---------------------------------------------------------------------------
// Spring constraint
// ---------------------------------------------------------------------------

interface Spring {
  bodyA: Body
  bodyB: Body
  restLength: number
  stiffness: number
  damping: number
}

// ---------------------------------------------------------------------------
// Collision manifold
// ---------------------------------------------------------------------------

interface Manifold {
  bodyA: Body
  bodyB: Body
  normal: Vec2       // from A to B
  penetration: number
  contactPoint: Vec2
}

// ---------------------------------------------------------------------------
// Collision detection
// ---------------------------------------------------------------------------

function getBoxCorners(b: Body): Vec2[] {
  const hw = b.halfW
  const hh = b.halfH
  const corners = [
    vec2(-hw, -hh),
    vec2(hw, -hh),
    vec2(hw, hh),
    vec2(-hw, hh),
  ]
  return corners.map(c => v2add(b.pos, v2rotate(c, b.angle)))
}

function getBoxAxes(b: Body): Vec2[] {
  const ax = v2rotate(vec2(1, 0), b.angle)
  const ay = v2rotate(vec2(0, 1), b.angle)
  return [ax, ay]
}

function projectOnAxis(corners: Vec2[], axis: Vec2): { min: number; max: number } {
  let min = Infinity
  let max = -Infinity
  for (const c of corners) {
    const proj = v2dot(c, axis)
    if (proj < min) min = proj
    if (proj > max) max = proj
  }
  return { min, max }
}

function collideCircleCircle(a: Body, b: Body): Manifold | null {
  const diff = v2sub(b.pos, a.pos)
  const distSq = v2lenSq(diff)
  const sumR = a.radius + b.radius
  if (distSq > sumR * sumR) return null
  const dist = Math.sqrt(distSq)
  if (dist < 1e-6) {
    // Bodies at same position, push apart
    return {
      bodyA: a,
      bodyB: b,
      normal: vec2(0, 1),
      penetration: sumR,
      contactPoint: a.pos,
    }
  }
  const normal = v2scale(diff, 1 / dist)
  const penetration = sumR - dist
  const contactPoint = v2add(a.pos, v2scale(normal, a.radius - penetration * 0.5))
  return { bodyA: a, bodyB: b, normal, penetration, contactPoint }
}

function collideCircleBox(circle: Body, box: Body): Manifold | null {
  // Transform circle center into box's local space
  const local = v2rotate(v2sub(circle.pos, box.pos), -box.angle)
  // Find closest point on box to circle center
  const closest = vec2(
    clamp(local.x, -box.halfW, box.halfW),
    clamp(local.y, -box.halfH, box.halfH)
  )
  const diff = v2sub(local, closest)
  const distSq = v2lenSq(diff)
  if (distSq > circle.radius * circle.radius) return null

  const dist = Math.sqrt(distSq)

  let localNormal: Vec2
  let penetration: number

  if (dist < 1e-6) {
    // Circle center is inside the box
    const dxLeft = local.x + box.halfW
    const dxRight = box.halfW - local.x
    const dyBottom = local.y + box.halfH
    const dyTop = box.halfH - local.y
    const minDist = Math.min(dxLeft, dxRight, dyBottom, dyTop)
    if (minDist === dxLeft) localNormal = vec2(-1, 0)
    else if (minDist === dxRight) localNormal = vec2(1, 0)
    else if (minDist === dyBottom) localNormal = vec2(0, -1)
    else localNormal = vec2(0, 1)
    penetration = circle.radius + minDist
  } else {
    localNormal = v2scale(diff, 1 / dist)
    penetration = circle.radius - dist
  }

  // Transform normal back to world space
  const worldNormal = v2rotate(localNormal, box.angle)
  const contactPoint = v2sub(circle.pos, v2scale(worldNormal, circle.radius - penetration * 0.5))

  return {
    bodyA: circle,
    bodyB: box,
    normal: worldNormal,
    penetration,
    contactPoint,
  }
}

function collideBoxBox(a: Body, b: Body): Manifold | null {
  const cornersA = getBoxCorners(a)
  const cornersB = getBoxCorners(b)
  const axesA = getBoxAxes(a)
  const axesB = getBoxAxes(b)
  const axes = [...axesA, ...axesB]

  let minOverlap = Infinity
  let minAxis = vec2(0, 0)

  for (const axis of axes) {
    const projA = projectOnAxis(cornersA, axis)
    const projB = projectOnAxis(cornersB, axis)
    const overlap = Math.min(projA.max - projB.min, projB.max - projA.min)
    if (overlap <= 0) return null
    if (overlap < minOverlap) {
      minOverlap = overlap
      minAxis = axis
    }
  }

  // Ensure normal points from A to B
  const d = v2sub(b.pos, a.pos)
  if (v2dot(d, minAxis) < 0) {
    minAxis = v2neg(minAxis)
  }

  // Approximate contact point: midpoint of deepest overlap
  const contactPoint = v2add(a.pos, v2scale(v2sub(b.pos, a.pos), 0.5))

  return {
    bodyA: a,
    bodyB: b,
    normal: minAxis,
    penetration: minOverlap,
    contactPoint,
  }
}

function detectCollision(a: Body, b: Body): Manifold | null {
  if (a.type === 'circle' && b.type === 'circle') {
    return collideCircleCircle(a, b)
  }
  if (a.type === 'circle' && b.type === 'box') {
    return collideCircleBox(a, b)
  }
  if (a.type === 'box' && b.type === 'circle') {
    const m = collideCircleBox(b, a)
    if (m) {
      m.bodyA = a
      m.bodyB = b
      m.normal = v2neg(m.normal)
    }
    return m
  }
  if (a.type === 'box' && b.type === 'box') {
    return collideBoxBox(a, b)
  }
  return null
}

// ---------------------------------------------------------------------------
// Collision response (impulse-based)
// ---------------------------------------------------------------------------

function resolveCollision(m: Manifold) {
  const a = m.bodyA
  const b = m.bodyB
  const n = m.normal

  // Positional correction (avoid sinking)
  const percent = 0.6
  const slop = 0.5
  const correctionMag = Math.max(m.penetration - slop, 0) / (a.invMass + b.invMass) * percent
  const correction = v2scale(n, correctionMag)
  a.pos = v2sub(a.pos, v2scale(correction, a.invMass))
  b.pos = v2add(b.pos, v2scale(correction, b.invMass))

  // Relative velocity at contact point
  const rA = v2sub(m.contactPoint, a.pos)
  const rB = v2sub(m.contactPoint, b.pos)
  const velA = v2add(a.vel, v2scale(v2perp(rA), a.angVel))
  const velB = v2add(b.vel, v2scale(v2perp(rB), b.angVel))
  const relVel = v2sub(velB, velA)
  const velAlongNormal = v2dot(relVel, n)

  // Separating already
  if (velAlongNormal > 0) return

  const e = Math.min(a.restitution, b.restitution)
  const rACrossN = v2cross(rA, n)
  const rBCrossN = v2cross(rB, n)

  const denominator = a.invMass + b.invMass +
    rACrossN * rACrossN * a.invInertia +
    rBCrossN * rBCrossN * b.invInertia

  const j = -(1 + e) * velAlongNormal / denominator
  const impulse = v2scale(n, j)

  a.vel = v2sub(a.vel, v2scale(impulse, a.invMass))
  b.vel = v2add(b.vel, v2scale(impulse, b.invMass))
  a.angVel -= v2cross(rA, impulse) * a.invInertia
  b.angVel += v2cross(rB, impulse) * b.invInertia

  // Friction impulse (tangent)
  const relVelAfter = v2sub(
    v2add(b.vel, v2scale(v2perp(rB), b.angVel)),
    v2add(a.vel, v2scale(v2perp(rA), a.angVel))
  )
  const tangent = v2norm(v2sub(relVelAfter, v2scale(n, v2dot(relVelAfter, n))))
  const velAlongTangent = v2dot(relVelAfter, tangent)
  if (Math.abs(velAlongTangent) < 1e-6) return

  const rACrossT = v2cross(rA, tangent)
  const rBCrossT = v2cross(rB, tangent)
  const tDenom = a.invMass + b.invMass +
    rACrossT * rACrossT * a.invInertia +
    rBCrossT * rBCrossT * b.invInertia

  let jt = -velAlongTangent / tDenom
  const mu = 0.3
  if (Math.abs(jt) > j * mu) {
    jt = j * mu * Math.sign(jt)
  }

  const frictionImpulse = v2scale(tangent, jt)
  a.vel = v2sub(a.vel, v2scale(frictionImpulse, a.invMass))
  b.vel = v2add(b.vel, v2scale(frictionImpulse, b.invMass))
  a.angVel -= v2cross(rA, frictionImpulse) * a.invInertia
  b.angVel += v2cross(rB, frictionImpulse) * b.invInertia
}

// ---------------------------------------------------------------------------
// Spatial hash grid for broad-phase collision detection
// ---------------------------------------------------------------------------

class SpatialGrid {
  private cellSize: number
  private cols: number
  private rows: number
  private grid: Int32Array
  private counts: Int32Array
  private cellCapacity: number

  constructor(width: number, height: number, cellSize: number, maxPerCell: number) {
    this.cellSize = cellSize
    this.cols = Math.max(1, Math.ceil(width / cellSize))
    this.rows = Math.max(1, Math.ceil(height / cellSize))
    this.cellCapacity = maxPerCell
    const totalCells = this.cols * this.rows
    this.grid = new Int32Array(totalCells * maxPerCell)
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
    this.cols = Math.max(1, Math.ceil(width / this.cellSize))
    this.rows = Math.max(1, Math.ceil(height / this.cellSize))
    const totalCells = this.cols * this.rows
    this.grid = new Int32Array(totalCells * this.cellCapacity)
    this.counts = new Int32Array(totalCells)
  }
}

// ---------------------------------------------------------------------------
// Pre-allocated scratch Vec2 for collision inner loop
// ---------------------------------------------------------------------------

const _scratch: Vec2 = { x: 0, y: 0 }

/** Mutable subtract: writes (a - b) into `out` and returns `out`. */
function v2subInto(out: Vec2, a: Vec2, b: Vec2): Vec2 {
  out.x = a.x - b.x
  out.y = a.y - b.y
  return out
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

export class PhysicsEngine {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D

  private bodies: Body[] = []
  private springs: Spring[] = []

  private tool: Tool = 'circle'
  private gravity: Vec2 = vec2(0, 400)
  private restitution = 0.5

  // Mouse state
  private mouse: Vec2 = vec2(0, 0)
  private mouseDown = false
  private dragStart: Vec2 | null = null
  private springAnchor: Body | null = null

  // Fixed timestep
  private readonly FIXED_DT = 1 / 60
  private accumulator = 0
  private lastTime = 0

  private animationId = 0
  private disposed = false
  private resizeObserver: ResizeObserver | null = null

  // Grid lines spacing
  private readonly GRID_SIZE = 40

  // Spatial grid for broad-phase collision detection
  private spatialGrid!: SpatialGrid
  private readonly SPATIAL_CELL_SIZE = 80

  // Body count change callback
  private bodyCountChangeCallback: ((count: number) => void) | null = null

  constructor(private container: HTMLDivElement) {
    this.canvas = document.createElement('canvas')
    this.canvas.style.width = '100%'
    this.canvas.style.height = '100%'
    this.canvas.style.display = 'block'
    container.appendChild(this.canvas)

    this.ctx = this.canvas.getContext('2d')!

    this.resizeCanvas()
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    this.spatialGrid = new SpatialGrid(w, h, this.SPATIAL_CELL_SIZE, 20)
    this.initListeners()
    this.lastTime = performance.now()
    this.animate()
  }

  // --- Public API ---

  setTool(tool: Tool) {
    this.tool = tool
    this.springAnchor = null
  }

  setGravity(strength: number) {
    this.gravity = vec2(0, strength)
  }

  setRestitution(value: number) {
    this.restitution = clamp(value, 0, 1)
    for (const b of this.bodies) b.restitution = this.restitution
  }

  clear() {
    this.bodies = []
    this.springs = []
    this.springAnchor = null
    this.notifyBodyCountChange()
  }

  get bodyCount(): number {
    return this.bodies.length
  }

  onBodyCountChange(callback: ((count: number) => void) | null) {
    this.bodyCountChangeCallback = callback
  }

  private notifyBodyCountChange() {
    if (this.bodyCountChangeCallback) {
      this.bodyCountChangeCallback(this.bodies.length)
    }
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

  // --- Resize ---

  private resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio, 2)
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    this.canvas.width = w * dpr
    this.canvas.height = h * dpr
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  // --- Input ---

  private getMousePos(e: MouseEvent | Touch): Vec2 {
    const rect = this.canvas.getBoundingClientRect()
    return vec2(e.clientX - rect.left, e.clientY - rect.top)
  }

  private bodyAtPoint(p: Vec2): Body | null {
    for (let i = this.bodies.length - 1; i >= 0; i--) {
      const b = this.bodies[i]
      if (b.type === 'circle') {
        if (v2lenSq(v2sub(p, b.pos)) <= b.radius * b.radius) return b
      } else {
        const local = v2rotate(v2sub(p, b.pos), -b.angle)
        if (Math.abs(local.x) <= b.halfW && Math.abs(local.y) <= b.halfH) return b
      }
    }
    return null
  }

  private onMouseDown = (e: MouseEvent) => {
    this.mouse = this.getMousePos(e)
    this.mouseDown = true

    if (this.tool === 'circle' || this.tool === 'box') {
      this.dragStart = { ...this.mouse }
    } else if (this.tool === 'spring') {
      const hit = this.bodyAtPoint(this.mouse)
      if (hit) {
        this.springAnchor = hit
      }
    } else if (this.tool === 'explosion') {
      this.applyExplosion(this.mouse)
    }
  }

  private onMouseMove = (e: MouseEvent) => {
    this.mouse = this.getMousePos(e)
  }

  private onMouseUp = (e: MouseEvent) => {
    this.mouse = this.getMousePos(e)
    this.mouseDown = false

    if (this.tool === 'circle' && this.dragStart) {
      const radius = v2len(v2sub(this.mouse, this.dragStart))
      if (radius > 3) {
        const body = createCircle(this.dragStart.x, this.dragStart.y, radius, this.restitution)
        this.bodies.push(body)
        this.notifyBodyCountChange()
      }
      this.dragStart = null
    } else if (this.tool === 'box' && this.dragStart) {
      const hw = Math.abs(this.mouse.x - this.dragStart.x) * 0.5
      const hh = Math.abs(this.mouse.y - this.dragStart.y) * 0.5
      if (hw > 3 && hh > 3) {
        const cx = (this.mouse.x + this.dragStart.x) * 0.5
        const cy = (this.mouse.y + this.dragStart.y) * 0.5
        const body = createBox(cx, cy, hw, hh, this.restitution)
        this.bodies.push(body)
        this.notifyBodyCountChange()
      }
      this.dragStart = null
    } else if (this.tool === 'spring' && this.springAnchor) {
      const hit = this.bodyAtPoint(this.mouse)
      if (hit && hit !== this.springAnchor) {
        const dist = v2len(v2sub(hit.pos, this.springAnchor.pos))
        this.springs.push({
          bodyA: this.springAnchor,
          bodyB: hit,
          restLength: dist,
          stiffness: 200,
          damping: 5,
        })
      }
      this.springAnchor = null
    }

    this.dragStart = null
  }

  private onTouchStart = (e: TouchEvent) => {
    e.preventDefault()
    const touch = e.touches[0]
    this.mouse = this.getMousePos(touch)
    this.mouseDown = true

    if (this.tool === 'circle' || this.tool === 'box') {
      this.dragStart = { ...this.mouse }
    } else if (this.tool === 'spring') {
      const hit = this.bodyAtPoint(this.mouse)
      if (hit) this.springAnchor = hit
    } else if (this.tool === 'explosion') {
      this.applyExplosion(this.mouse)
    }
  }

  private onTouchMove = (e: TouchEvent) => {
    e.preventDefault()
    this.mouse = this.getMousePos(e.touches[0])
  }

  private onTouchEnd = () => {
    // Replicate mouseUp logic
    this.mouseDown = false
    if (this.tool === 'circle' && this.dragStart) {
      const radius = v2len(v2sub(this.mouse, this.dragStart))
      if (radius > 3) {
        this.bodies.push(createCircle(this.dragStart.x, this.dragStart.y, radius, this.restitution))
        this.notifyBodyCountChange()
      }
      this.dragStart = null
    } else if (this.tool === 'box' && this.dragStart) {
      const hw = Math.abs(this.mouse.x - this.dragStart.x) * 0.5
      const hh = Math.abs(this.mouse.y - this.dragStart.y) * 0.5
      if (hw > 3 && hh > 3) {
        const cx = (this.mouse.x + this.dragStart.x) * 0.5
        const cy = (this.mouse.y + this.dragStart.y) * 0.5
        this.bodies.push(createBox(cx, cy, hw, hh, this.restitution))
        this.notifyBodyCountChange()
      }
      this.dragStart = null
    } else if (this.tool === 'spring' && this.springAnchor) {
      const hit = this.bodyAtPoint(this.mouse)
      if (hit && hit !== this.springAnchor) {
        const dist = v2len(v2sub(hit.pos, this.springAnchor.pos))
        this.springs.push({
          bodyA: this.springAnchor,
          bodyB: hit,
          restLength: dist,
          stiffness: 200,
          damping: 5,
        })
      }
      this.springAnchor = null
    }
  }

  private initListeners() {
    this.canvas.addEventListener('mousedown', this.onMouseDown)
    this.canvas.addEventListener('mousemove', this.onMouseMove)
    window.addEventListener('mouseup', this.onMouseUp)
    this.canvas.addEventListener('touchstart', this.onTouchStart, { passive: false })
    this.canvas.addEventListener('touchmove', this.onTouchMove, { passive: false })
    window.addEventListener('touchend', this.onTouchEnd)

    this.resizeObserver = new ResizeObserver(() => {
      if (!this.disposed) {
        this.resizeCanvas()
        this.spatialGrid.resize(this.container.clientWidth, this.container.clientHeight)
      }
    })
    this.resizeObserver.observe(this.container)
  }

  private removeListeners() {
    this.canvas.removeEventListener('mousedown', this.onMouseDown)
    this.canvas.removeEventListener('mousemove', this.onMouseMove)
    window.removeEventListener('mouseup', this.onMouseUp)
    this.canvas.removeEventListener('touchstart', this.onTouchStart)
    this.canvas.removeEventListener('touchmove', this.onTouchMove)
    window.removeEventListener('touchend', this.onTouchEnd)
  }

  // --- Tools ---

  private applyGravityGun(dt: number) {
    if (!this.mouseDown || this.tool !== 'gravity') return
    const range = 250
    const strength = 8000
    for (const b of this.bodies) {
      const diff = v2sub(this.mouse, b.pos)
      const dist = v2len(diff)
      if (dist < range && dist > 1) {
        const force = v2scale(v2norm(diff), strength * dt / Math.max(dist, 30))
        b.vel = v2add(b.vel, v2scale(force, b.invMass))
      }
    }
  }

  private applyExplosion(center: Vec2) {
    const strength = 50000
    const range = 300
    for (const b of this.bodies) {
      const diff = v2sub(b.pos, center)
      const dist = v2len(diff)
      if (dist < range && dist > 1) {
        const force = v2scale(v2norm(diff), strength / Math.max(dist, 20))
        b.vel = v2add(b.vel, v2scale(force, b.invMass))
      }
    }
  }

  // --- Physics step ---

  private physicsStep(dt: number) {
    const w = this.container.clientWidth
    const h = this.container.clientHeight

    // Gravity gun
    this.applyGravityGun(dt)

    // Apply gravity
    for (const b of this.bodies) {
      if (b.isStatic) continue
      b.vel = v2add(b.vel, v2scale(this.gravity, dt))
    }

    // Apply spring forces
    for (const s of this.springs) {
      const diff = v2sub(s.bodyB.pos, s.bodyA.pos)
      const dist = v2len(diff)
      if (dist < 1e-6) continue
      const n = v2scale(diff, 1 / dist)
      const displacement = dist - s.restLength
      const relVel = v2sub(s.bodyB.vel, s.bodyA.vel)
      const dampingForce = v2dot(relVel, n) * s.damping
      const forceMag = displacement * s.stiffness + dampingForce
      const force = v2scale(n, forceMag)

      s.bodyA.vel = v2add(s.bodyA.vel, v2scale(force, s.bodyA.invMass * dt))
      s.bodyB.vel = v2sub(s.bodyB.vel, v2scale(force, s.bodyB.invMass * dt))
    }

    // Integrate positions
    for (const b of this.bodies) {
      if (b.isStatic) continue
      b.pos = v2add(b.pos, v2scale(b.vel, dt))
      b.angle += b.angVel * dt
      // Angular damping
      b.angVel *= 0.998
      // Linear damping
      b.vel = v2scale(b.vel, 0.999)
    }

    // Rebuild spatial grid and detect collisions via broad phase
    const bodies = this.bodies
    const n = bodies.length
    this.spatialGrid.clear()

    // Find max radius across all bodies for query range
    let maxRadius = 0
    for (let i = 0; i < n; i++) {
      if (bodies[i].radius > maxRadius) maxRadius = bodies[i].radius
      this.spatialGrid.insert(i, bodies[i].pos.x, bodies[i].pos.y)
    }

    // Query radius: each body needs to find neighbors within (its radius + maxRadius)
    // We use maxRadius * 2 as the query range to cover the worst case pair
    const queryRange = maxRadius * 2 * 1.1 // 1.1 factor matches the old 1.2 on squared distance

    for (let i = 0; i < n; i++) {
      const a = bodies[i]
      this.spatialGrid.query(a.pos.x, a.pos.y, queryRange, (j: number) => {
        if (j <= i) return // avoid duplicate pairs and self
        const b = bodies[j]
        // Narrow broad phase: bounding sphere check with pre-allocated scratch
        v2subInto(_scratch, a.pos, b.pos)
        const distSq = _scratch.x * _scratch.x + _scratch.y * _scratch.y
        const sumR = a.radius + b.radius
        if (distSq > sumR * sumR * 1.2) return

        const manifold = detectCollision(a, b)
        if (manifold) {
          resolveCollision(manifold)
        }
      })
    }

    // Wall collisions
    for (const b of bodies) {
      this.resolveWall(b, w, h)
    }

    // Remove bodies that fell far off-screen
    const prevCount = this.bodies.length
    this.bodies = this.bodies.filter(b =>
      b.pos.y < h + 500 && b.pos.y > -500 && b.pos.x > -500 && b.pos.x < w + 500
    )
    if (this.bodies.length !== prevCount) {
      this.notifyBodyCountChange()
    }

    // Remove springs whose bodies no longer exist
    const bodySet = new Set(this.bodies)
    this.springs = this.springs.filter(s => bodySet.has(s.bodyA) && bodySet.has(s.bodyB))
  }

  private resolveWall(b: Body, w: number, h: number) {
    const bounce = b.restitution

    if (b.type === 'circle') {
      // Bottom
      if (b.pos.y + b.radius > h) {
        b.pos.y = h - b.radius
        b.vel.y = -Math.abs(b.vel.y) * bounce
        b.angVel += b.vel.x * 0.01
      }
      // Top
      if (b.pos.y - b.radius < 0) {
        b.pos.y = b.radius
        b.vel.y = Math.abs(b.vel.y) * bounce
      }
      // Right
      if (b.pos.x + b.radius > w) {
        b.pos.x = w - b.radius
        b.vel.x = -Math.abs(b.vel.x) * bounce
      }
      // Left
      if (b.pos.x - b.radius < 0) {
        b.pos.x = b.radius
        b.vel.x = Math.abs(b.vel.x) * bounce
      }
    } else {
      // Use bounding radius for approximate wall collision
      const r = b.radius
      if (b.pos.y + r > h) {
        b.pos.y = h - r
        b.vel.y = -Math.abs(b.vel.y) * bounce
        b.angVel += b.vel.x * 0.005
      }
      if (b.pos.y - r < 0) {
        b.pos.y = r
        b.vel.y = Math.abs(b.vel.y) * bounce
      }
      if (b.pos.x + r > w) {
        b.pos.x = w - r
        b.vel.x = -Math.abs(b.vel.x) * bounce
      }
      if (b.pos.x - r < 0) {
        b.pos.x = r
        b.vel.x = Math.abs(b.vel.x) * bounce
      }
    }
  }

  // --- Rendering ---

  private draw() {
    const ctx = this.ctx
    const w = this.container.clientWidth
    const h = this.container.clientHeight

    // Clear
    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, w, h)

    // Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)'
    ctx.lineWidth = 1
    ctx.beginPath()
    for (let x = 0; x < w; x += this.GRID_SIZE) {
      ctx.moveTo(x, 0)
      ctx.lineTo(x, h)
    }
    for (let y = 0; y < h; y += this.GRID_SIZE) {
      ctx.moveTo(0, y)
      ctx.lineTo(w, y)
    }
    ctx.stroke()

    // Springs
    ctx.setLineDash([6, 4])
    ctx.lineWidth = 1.5
    for (const s of this.springs) {
      const stress = Math.abs(v2len(v2sub(s.bodyB.pos, s.bodyA.pos)) - s.restLength) / s.restLength
      const r = Math.min(255, 120 + stress * 400)
      const g = Math.max(0, 180 - stress * 300)
      ctx.strokeStyle = `rgba(${r | 0}, ${g | 0}, 120, 0.7)`
      ctx.beginPath()
      ctx.moveTo(s.bodyA.pos.x, s.bodyA.pos.y)
      ctx.lineTo(s.bodyB.pos.x, s.bodyB.pos.y)
      ctx.stroke()
    }
    ctx.setLineDash([])

    // Bodies
    for (const b of this.bodies) {
      ctx.save()
      ctx.translate(b.pos.x, b.pos.y)
      ctx.rotate(b.angle)

      if (b.type === 'circle') {
        ctx.beginPath()
        ctx.arc(0, 0, b.radius, 0, Math.PI * 2)
        ctx.fillStyle = b.color
        ctx.globalAlpha = 0.85
        ctx.fill()
        ctx.globalAlpha = 1
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)'
        ctx.lineWidth = 1
        ctx.stroke()

        // Rotation indicator
        ctx.beginPath()
        ctx.moveTo(0, 0)
        ctx.lineTo(b.radius * 0.8, 0)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
        ctx.lineWidth = 1.5
        ctx.stroke()
      } else {
        ctx.fillStyle = b.color
        ctx.globalAlpha = 0.85
        ctx.fillRect(-b.halfW, -b.halfH, b.halfW * 2, b.halfH * 2)
        ctx.globalAlpha = 1
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)'
        ctx.lineWidth = 1
        ctx.strokeRect(-b.halfW, -b.halfH, b.halfW * 2, b.halfH * 2)

        // Corner dot
        ctx.fillStyle = 'rgba(255, 255, 255, 0.3)'
        ctx.beginPath()
        ctx.arc(b.halfW - 4, -b.halfH + 4, 2, 0, Math.PI * 2)
        ctx.fill()
      }

      ctx.restore()
    }

    // Draw tool preview
    this.drawToolPreview(ctx)

    // Spring anchor line
    if (this.tool === 'spring' && this.springAnchor) {
      ctx.strokeStyle = 'rgba(255, 180, 60, 0.6)'
      ctx.lineWidth = 2
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(this.springAnchor.pos.x, this.springAnchor.pos.y)
      ctx.lineTo(this.mouse.x, this.mouse.y)
      ctx.stroke()
      ctx.setLineDash([])
    }

    // Gravity gun indicator
    if (this.tool === 'gravity' && this.mouseDown) {
      ctx.strokeStyle = 'rgba(100, 200, 255, 0.25)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.arc(this.mouse.x, this.mouse.y, 250, 0, Math.PI * 2)
      ctx.stroke()

      ctx.fillStyle = 'rgba(100, 200, 255, 0.4)'
      ctx.beginPath()
      ctx.arc(this.mouse.x, this.mouse.y, 6, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  private drawToolPreview(ctx: CanvasRenderingContext2D) {
    if (!this.dragStart || !this.mouseDown) return

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'
    ctx.lineWidth = 1
    ctx.setLineDash([4, 4])

    if (this.tool === 'circle') {
      const r = v2len(v2sub(this.mouse, this.dragStart))
      ctx.beginPath()
      ctx.arc(this.dragStart.x, this.dragStart.y, r, 0, Math.PI * 2)
      ctx.stroke()
    } else if (this.tool === 'box') {
      const x = Math.min(this.dragStart.x, this.mouse.x)
      const y = Math.min(this.dragStart.y, this.mouse.y)
      const w = Math.abs(this.mouse.x - this.dragStart.x)
      const h = Math.abs(this.mouse.y - this.dragStart.y)
      ctx.strokeRect(x, y, w, h)
    }

    ctx.setLineDash([])
  }

  // --- Main loop ---

  private animate = () => {
    if (this.disposed) return
    this.animationId = requestAnimationFrame(this.animate)

    const now = performance.now()
    const frameDt = Math.min((now - this.lastTime) * 0.001, 0.1)
    this.lastTime = now

    this.accumulator += frameDt
    while (this.accumulator >= this.FIXED_DT) {
      this.physicsStep(this.FIXED_DT)
      this.accumulator -= this.FIXED_DT
    }

    this.draw()
  }
}
