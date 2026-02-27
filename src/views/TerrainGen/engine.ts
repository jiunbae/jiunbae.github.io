/**
 * Three.js terrain generator with multi-octave simplex noise,
 * hydraulic erosion, biome coloring, and water plane.
 */

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

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

// ── Biome color palette ─────────────────────────────────────────────

// Pre-allocated Color instance reused by getBiomeColor (H-3 fix: avoids 65K allocations per rebuild)
const _biomeColor = new THREE.Color()

function getBiomeColor(height: number, moisture: number): THREE.Color {
  // height: 0..1 normalized, moisture: 0..1

  if (height < 0.28) {
    return _biomeColor.setRGB(0.05, 0.1, 0.35)
  }
  if (height < 0.35) {
    const t = (height - 0.28) / 0.07
    return _biomeColor.setRGB(0.1 + t * 0.15, 0.25 + t * 0.15, 0.5 + t * 0.1)
  }
  if (height < 0.38) {
    return _biomeColor.setRGB(0.76, 0.7, 0.5)
  }
  if (height < 0.55) {
    if (moisture > 0.5) {
      return _biomeColor.setRGB(0.1, 0.3, 0.08)
    }
    return _biomeColor.setRGB(0.25, 0.5, 0.15)
  }
  if (height < 0.7) {
    if (moisture > 0.4) {
      return _biomeColor.setRGB(0.08, 0.25, 0.06)
    }
    const t = (height - 0.55) / 0.15
    return _biomeColor.setRGB(0.25 + t * 0.2, 0.5 - t * 0.25, 0.15 - t * 0.05)
  }
  if (height < 0.85) {
    const t = (height - 0.7) / 0.15
    return _biomeColor.setRGB(0.4 + t * 0.1, 0.38 + t * 0.1, 0.35 + t * 0.1)
  }
  const t = Math.min((height - 0.85) / 0.15, 1.0)
  return _biomeColor.setRGB(0.7 + t * 0.2, 0.72 + t * 0.2, 0.78 + t * 0.17)
}

// ── Terrain Engine ──────────────────────────────────────────────────

export class TerrainEngine {
  private scene: THREE.Scene
  private camera: THREE.PerspectiveCamera
  private renderer: THREE.WebGLRenderer
  private controls: OrbitControls
  private terrain: THREE.Mesh | null = null
  private waterPlane: THREE.Mesh | null = null

  private animId = 0
  private disposed = false
  private resizeObserver: ResizeObserver | null = null

  private currentSeed: number
  private rng!: () => number
  private noise!: SimplexNoise2D

  private noiseScale = 2.0
  private octaveCount = 5
  private erosionSteps = 500
  private waterLevelValue = 0.35

  private static readonly GRID_SIZE = 256

  constructor(private container: HTMLDivElement) {
    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.setSize(container.clientWidth, container.clientHeight)
    this.renderer.setClearColor(0x1a1a2e)
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap
    container.appendChild(this.renderer.domElement)

    this.renderer.domElement.addEventListener('webglcontextlost', this.handleContextLost)
    this.renderer.domElement.addEventListener('webglcontextrestored', this.handleContextRestored)

    // Scene
    this.scene = new THREE.Scene()
    this.scene.fog = new THREE.FogExp2(0x1a1a2e, 0.015)

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      55,
      container.clientWidth / container.clientHeight,
      0.1,
      200
    )
    this.camera.position.set(35, 25, 35)

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.05
    this.controls.autoRotate = true
    this.controls.autoRotateSpeed = 0.5
    this.controls.maxPolarAngle = Math.PI * 0.48
    this.controls.minDistance = 15
    this.controls.maxDistance = 80
    this.controls.target.set(0, 0, 0)

    // Lights
    const ambient = new THREE.AmbientLight(0x404060, 0.6)
    this.scene.add(ambient)

    const directional = new THREE.DirectionalLight(0xfff4e0, 1.2)
    directional.position.set(30, 40, 20)
    directional.castShadow = true
    directional.shadow.mapSize.width = 1024
    directional.shadow.mapSize.height = 1024
    directional.shadow.camera.near = 1
    directional.shadow.camera.far = 100
    directional.shadow.camera.left = -40
    directional.shadow.camera.right = 40
    directional.shadow.camera.top = 40
    directional.shadow.camera.bottom = -40
    this.scene.add(directional)

    const hemiLight = new THREE.HemisphereLight(0x87ceeb, 0x362d1b, 0.3)
    this.scene.add(hemiLight)

    // Generate initial terrain
    this.currentSeed = Math.floor(Math.random() * 2147483647)
    this.buildTerrain()

    // Resize
    this.resizeObserver = new ResizeObserver(() => {
      if (!this.disposed) this.handleResize()
    })
    this.resizeObserver.observe(container)

    this.animate()
  }

  // ── Public API ────────────────────────────────────────────────

  generate(seed?: number) {
    this.currentSeed = seed ?? Math.floor(Math.random() * 2147483647)
    this.buildTerrain()
  }

  setNoiseScale(v: number) {
    this.noiseScale = Math.max(0.5, Math.min(5, v))
    this.buildTerrain()
  }

  setOctaves(v: number) {
    this.octaveCount = Math.max(1, Math.min(8, Math.round(v)))
    this.buildTerrain()
  }

  setErosionSteps(v: number) {
    this.erosionSteps = Math.max(0, Math.min(5000, Math.round(v)))
    this.buildTerrain()
  }

  setWaterLevel(v: number) {
    this.waterLevelValue = Math.max(0, Math.min(1, v))
    this.updateWaterPlane()
    this.updateTerrainColors()
  }

  get autoRotate(): boolean {
    return this.controls.autoRotate
  }

  set autoRotate(value: boolean) {
    this.controls.autoRotate = value
  }

  get seed(): number {
    return this.currentSeed
  }

  dispose() {
    this.disposed = true
    cancelAnimationFrame(this.animId)

    if (this.resizeObserver) {
      this.resizeObserver.disconnect()
      this.resizeObserver = null
    }

    this.controls.dispose()

    // Dispose scene objects
    this.scene.traverse(obj => {
      if (obj instanceof THREE.Mesh) {
        obj.geometry.dispose()
        if (Array.isArray(obj.material)) {
          obj.material.forEach(m => m.dispose())
        } else {
          obj.material.dispose()
        }
      }
    })

    this.renderer.domElement.removeEventListener('webglcontextlost', this.handleContextLost)
    this.renderer.domElement.removeEventListener('webglcontextrestored', this.handleContextRestored)
    this.renderer.dispose()
    if (this.renderer.domElement.parentElement) {
      this.renderer.domElement.parentElement.removeChild(this.renderer.domElement)
    }
  }

  // ── Terrain generation ────────────────────────────────────────

  private buildTerrain() {
    this.rng = mulberry32(this.currentSeed)
    this.noise = new SimplexNoise2D(this.rng)

    const size = TerrainEngine.GRID_SIZE
    const heightMap = this.generateHeightMap(size)

    // Apply erosion
    if (this.erosionSteps > 0) {
      this.erode(heightMap, size, this.erosionSteps)
    }

    // Build geometry
    this.buildGeometry(heightMap, size)
    this.updateWaterPlane()
  }

  private generateHeightMap(size: number): Float32Array {
    const heightMap = new Float32Array((size + 1) * (size + 1))
    const scale = this.noiseScale
    const octaves = this.octaveCount

    for (let z = 0; z <= size; z++) {
      for (let x = 0; x <= size; x++) {
        const nx = (x / size - 0.5) * scale
        const nz = (z / size - 0.5) * scale

        let height = 0
        let amplitude = 1
        let frequency = 1
        let maxAmp = 0

        for (let o = 0; o < octaves; o++) {
          height += this.noise.noise(nx * frequency, nz * frequency) * amplitude
          maxAmp += amplitude
          amplitude *= 0.5
          frequency *= 2.1
        }

        height /= maxAmp
        // Normalize to 0..1
        height = (height + 1) * 0.5

        // Island falloff: push edges down
        const dx = (x / size - 0.5) * 2
        const dz = (z / size - 0.5) * 2
        const distFromCenter = Math.sqrt(dx * dx + dz * dz)
        const falloff = Math.max(0, 1 - distFromCenter * distFromCenter)
        height *= falloff

        heightMap[z * (size + 1) + x] = height
      }
    }

    return heightMap
  }

  // ── Hydraulic erosion (simplified droplet-based) ──────────────

  private erode(heightMap: Float32Array, size: number, iterations: number) {
    const stride = size + 1
    const rng = this.rng

    for (let iter = 0; iter < iterations; iter++) {
      // Random droplet position
      let x = rng() * (size - 2) + 1
      let y = rng() * (size - 2) + 1
      let dx = 0
      let dy = 0
      let speed = 0
      let water = 1
      let sediment = 0

      const inertia = 0.05
      const capacity = 4.0
      const deposition = 0.3
      const erosion = 0.3
      const evaporation = 0.01
      const gravity = 4.0
      const maxSteps = 80

      for (let step = 0; step < maxSteps; step++) {
        const ix = Math.floor(x)
        const iy = Math.floor(y)

        if (ix < 1 || ix >= size - 1 || iy < 1 || iy >= size - 1) break

        // Bilinear interpolation offset
        const fx = x - ix
        const fy = y - iy

        // Heights at corners
        const h00 = heightMap[iy * stride + ix]
        const h10 = heightMap[iy * stride + ix + 1]
        const h01 = heightMap[(iy + 1) * stride + ix]
        const h11 = heightMap[(iy + 1) * stride + ix + 1]

        // Gradient
        const gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
        const gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx

        // Update direction with inertia
        dx = dx * inertia - gx * (1 - inertia)
        dy = dy * inertia - gy * (1 - inertia)

        // Normalize direction
        const len = Math.sqrt(dx * dx + dy * dy)
        if (len < 0.0001) {
          dx = rng() * 2 - 1
          dy = rng() * 2 - 1
        } else {
          dx /= len
          dy /= len
        }

        // Move
        const newX = x + dx
        const newY = y + dy

        if (newX < 1 || newX >= size - 1 || newY < 1 || newY >= size - 1) break

        // Height at old and new positions
        const oldH = h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) + h01 * (1 - fx) * fy + h11 * fx * fy

        const nix = Math.floor(newX)
        const niy = Math.floor(newY)
        const nfx = newX - nix
        const nfy = newY - niy
        const nh00 = heightMap[niy * stride + nix]
        const nh10 = heightMap[niy * stride + nix + 1]
        const nh01 = heightMap[(niy + 1) * stride + nix]
        const nh11 = heightMap[(niy + 1) * stride + nix + 1]
        const newH = nh00 * (1 - nfx) * (1 - nfy) + nh10 * nfx * (1 - nfy) + nh01 * (1 - nfx) * nfy + nh11 * nfx * nfy

        const heightDiff = newH - oldH

        // Sediment capacity
        const c = Math.max(-heightDiff, 0.01) * speed * water * capacity

        if (sediment > c || heightDiff > 0) {
          // Deposit
          const depositAmount = heightDiff > 0
            ? Math.min(sediment, heightDiff)
            : (sediment - c) * deposition

          sediment -= depositAmount

          // Distribute deposit to the 4 surrounding cells
          heightMap[iy * stride + ix] += depositAmount * (1 - fx) * (1 - fy)
          heightMap[iy * stride + ix + 1] += depositAmount * fx * (1 - fy)
          heightMap[(iy + 1) * stride + ix] += depositAmount * (1 - fx) * fy
          heightMap[(iy + 1) * stride + ix + 1] += depositAmount * fx * fy
        } else {
          // Erode
          const erodeAmount = Math.min((c - sediment) * erosion, -heightDiff)

          heightMap[iy * stride + ix] -= erodeAmount * (1 - fx) * (1 - fy)
          heightMap[iy * stride + ix + 1] -= erodeAmount * fx * (1 - fy)
          heightMap[(iy + 1) * stride + ix] -= erodeAmount * (1 - fx) * fy
          heightMap[(iy + 1) * stride + ix + 1] -= erodeAmount * fx * fy

          sediment += erodeAmount
        }

        // Update speed and water
        speed = Math.sqrt(Math.max(0, speed * speed + heightDiff * gravity))
        water *= (1 - evaporation)

        x = newX
        y = newY

        if (water < 0.001) break
      }
    }
  }

  // ── Geometry construction ─────────────────────────────────────

  private buildGeometry(heightMap: Float32Array, size: number) {
    // Remove old terrain
    if (this.terrain) {
      this.scene.remove(this.terrain)
      this.terrain.geometry.dispose()
      ;(this.terrain.material as THREE.Material).dispose()
    }

    const terrainSize = 50
    const geometry = new THREE.PlaneGeometry(terrainSize, terrainSize, size, size)
    geometry.rotateX(-Math.PI / 2)

    const positions = geometry.attributes.position
    const stride = size + 1
    const heightScale = 15
    const colors = new Float32Array(positions.count * 3)

    // Second noise for moisture
    const moistureRng = mulberry32(this.currentSeed + 12345)
    const moistureNoise = new SimplexNoise2D(moistureRng)

    for (let i = 0; i < positions.count; i++) {
      const x = i % stride
      const z = Math.floor(i / stride)
      const h = heightMap[z * stride + x]

      // Set Y position
      positions.setY(i, h * heightScale)

      // Moisture for biome
      const nx = (x / size - 0.5) * 3
      const nz = (z / size - 0.5) * 3
      const moisture = (moistureNoise.noise(nx, nz) + 1) * 0.5

      // Biome color
      const color = getBiomeColor(h, moisture)
      colors[i * 3] = color.r
      colors[i * 3 + 1] = color.g
      colors[i * 3 + 2] = color.b
    }

    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geometry.computeVertexNormals()

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.85,
      metalness: 0.05,
      flatShading: false
    })

    this.terrain = new THREE.Mesh(geometry, material)
    this.terrain.castShadow = true
    this.terrain.receiveShadow = true
    this.scene.add(this.terrain)
  }

  private updateTerrainColors() {
    if (!this.terrain) return

    const geometry = this.terrain.geometry as THREE.BufferGeometry
    const positions = geometry.attributes.position
    const colorAttr = geometry.attributes.color
    if (!colorAttr) return

    const size = TerrainEngine.GRID_SIZE
    const stride = size + 1
    const heightScale = 15

    const moistureRng = mulberry32(this.currentSeed + 12345)
    const moistureNoise = new SimplexNoise2D(moistureRng)

    for (let i = 0; i < positions.count; i++) {
      const x = i % stride
      const z = Math.floor(i / stride)
      const h = positions.getY(i) / heightScale

      const nx = (x / size - 0.5) * 3
      const nz = (z / size - 0.5) * 3
      const moisture = (moistureNoise.noise(nx, nz) + 1) * 0.5

      const color = getBiomeColor(h, moisture)
      colorAttr.setXYZ(i, color.r, color.g, color.b)
    }

    colorAttr.needsUpdate = true
  }

  private updateWaterPlane() {
    const heightScale = 15
    const waterY = this.waterLevelValue * heightScale

    if (this.waterPlane) {
      this.waterPlane.position.y = waterY
    } else {
      const waterGeom = new THREE.PlaneGeometry(55, 55)
      waterGeom.rotateX(-Math.PI / 2)

      const waterMat = new THREE.MeshStandardMaterial({
        color: 0x1a6eb5,
        transparent: true,
        opacity: 0.55,
        roughness: 0.1,
        metalness: 0.3,
        side: THREE.DoubleSide
      })

      this.waterPlane = new THREE.Mesh(waterGeom, waterMat)
      this.waterPlane.position.y = waterY
      this.waterPlane.receiveShadow = true
      this.scene.add(this.waterPlane)
    }
  }

  // ── Resize / animate ──────────────────────────────────────────

  private handleResize() {
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h)
  }

  private handleContextLost = (e: Event) => {
    e.preventDefault()
    cancelAnimationFrame(this.animId)
    const overlay = document.createElement('div')
    overlay.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.8);color:#fff;font:14px/1.5 sans-serif;cursor:pointer;z-index:100'
    overlay.textContent = 'WebGL context lost — click to reload'
    overlay.addEventListener('click', () => location.reload())
    this.container.style.position = 'relative'
    this.container.appendChild(overlay)
  }

  private handleContextRestored = () => {
    location.reload()
  }

  private animate = () => {
    if (this.disposed) return
    this.animId = requestAnimationFrame(this.animate)
    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }
}
