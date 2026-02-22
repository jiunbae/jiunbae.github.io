import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'

/* -------------------------------------------------------------------------- */
/*  Constants                                                                  */
/* -------------------------------------------------------------------------- */

const PARTICLE_COUNT = 80_000
const NUM_ARMS = 4
const ARM_SPREAD = 0.45           // gaussian scatter width (radians)
const GALAXY_RADIUS = 40
const GALAXY_THICKNESS = 1.2
const CORE_RADIUS = 4
const BASE_ORBIT_SPEED = 0.08     // base angular velocity
const MAX_ATTRACTORS = 12
const ATTRACTOR_STRENGTH = 120
const ATTRACTOR_SOFTENING = 2.0   // prevent infinite force at zero distance

/* -------------------------------------------------------------------------- */
/*  Shaders                                                                    */
/* -------------------------------------------------------------------------- */

const vertexShader = /* glsl */ `
  uniform float uTime;
  uniform float uPixelRatio;
  uniform vec3  uAttractors[${MAX_ATTRACTORS}];
  uniform int   uAttractorCount;
  uniform float uAttractorStrength;
  uniform float uAttractorSoftening;

  attribute vec3  aVelocity;
  attribute float aMass;
  attribute float aOrbitSpeed;
  attribute float aPhase;
  attribute float aRadius;

  varying vec3 vColor;
  varying float vAlpha;

  /* Convert speed magnitude to a star colour (black-body inspired) */
  vec3 speedToColor(float speed) {
    // Slow (red/orange) -> medium (yellow/white) -> fast (blue/white)
    float t = clamp(speed * 3.0, 0.0, 1.0);
    vec3 cold = vec3(1.0, 0.35, 0.15);   // cool red-orange
    vec3 warm = vec3(1.0, 0.85, 0.5);    // warm yellow
    vec3 hot  = vec3(0.6, 0.75, 1.0);    // hot blue-white
    vec3 col = mix(cold, warm, smoothstep(0.0, 0.4, t));
    col = mix(col, hot, smoothstep(0.4, 1.0, t));
    return col;
  }

  void main() {
    /* ---- orbital motion ---- */
    float angle = aPhase + uTime * aOrbitSpeed;
    float r = aRadius;

    /* base spiral position */
    vec3 pos;
    pos.x = cos(angle) * r;
    pos.z = sin(angle) * r;
    pos.y = aVelocity.y; // stored vertical offset

    /* add wobble from initial velocity */
    pos.x += sin(uTime * aVelocity.x * 0.5 + aPhase) * 0.3;
    pos.z += cos(uTime * aVelocity.z * 0.5 + aPhase) * 0.3;

    /* ---- gravitational attractors ---- */
    vec3 totalAccel = vec3(0.0);
    for (int i = 0; i < ${MAX_ATTRACTORS}; i++) {
      if (i >= uAttractorCount) break;
      vec3 toAttractor = uAttractors[i] - pos;
      float dist2 = dot(toAttractor, toAttractor) + uAttractorSoftening;
      float force = uAttractorStrength / dist2;
      totalAccel += normalize(toAttractor) * force;
    }
    pos += totalAccel * 0.02;

    /* ---- velocity magnitude for colouring ---- */
    float tangentialSpeed = aOrbitSpeed * r;
    float attractorPull = length(totalAccel);
    float speed = tangentialSpeed + attractorPull * 0.15;

    vColor = speedToColor(speed);

    /* brighter in core */
    float coreBrightness = smoothstep(GALAXY_RADIUS, 0.0, r) * 0.4;
    vColor += coreBrightness;

    /* alpha: slightly fade outer particles */
    vAlpha = 0.7 + 0.3 * smoothstep(GALAXY_RADIUS, 0.0, r);

    /* ---- projection ---- */
    vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mvPos;

    /* size: bigger when closer, smaller far away; mass affects size */
    float baseSize = aMass * 1.8 + 0.8;
    gl_PointSize = baseSize * uPixelRatio * (300.0 / -mvPos.z);
    gl_PointSize = clamp(gl_PointSize, 0.5, 64.0);
  }
`

const fragmentShader = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    /* soft glow point-sprite */
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;

    /* gaussian-ish glow */
    float intensity = exp(-d * d * 8.0);

    /* add a bright core with soft halo */
    float core = smoothstep(0.15, 0.0, d);
    vec3 col = vColor * intensity + vec3(1.0) * core * 0.3;

    float alpha = intensity * vAlpha;
    gl_FragColor = vec4(col, alpha);
  }
`

/* -------------------------------------------------------------------------- */
/*  Scene class                                                                */
/* -------------------------------------------------------------------------- */

export class ParticleGalaxyScene {
  private renderer!: THREE.WebGLRenderer
  private scene!: THREE.Scene
  private camera!: THREE.PerspectiveCamera
  private controls!: OrbitControls
  private composer!: EffectComposer
  private bloomPass!: UnrealBloomPass
  private points!: THREE.Points
  private material!: THREE.ShaderMaterial
  private raycaster = new THREE.Raycaster()
  private galaxyPlane!: THREE.Mesh
  private clock = new THREE.Clock()
  private animationId = 0
  private disposed = false

  private attractorPositions: THREE.Vector3[] = []

  /* ---- public API ---- */

  get autoRotate(): boolean {
    return this.controls?.autoRotate ?? true
  }
  set autoRotate(v: boolean) {
    if (this.controls) this.controls.autoRotate = v
  }

  get bloomStrength(): number {
    return this.bloomPass?.strength ?? 1.5
  }
  set bloomStrength(v: number) {
    if (this.bloomPass) this.bloomPass.strength = v
  }

  get particleCount(): number {
    return PARTICLE_COUNT
  }

  get attractorCount(): number {
    return this.attractorPositions.length
  }

  addAttractor(ndcX: number, ndcY: number) {
    if (this.attractorPositions.length >= MAX_ATTRACTORS) return

    const mouse = new THREE.Vector2(ndcX, ndcY)
    this.raycaster.setFromCamera(mouse, this.camera)
    const hits = this.raycaster.intersectObject(this.galaxyPlane)
    if (hits.length === 0) return

    const p = hits[0].point
    this.attractorPositions.push(p.clone())
    this.syncAttractorUniforms()
  }

  clearAttractors() {
    this.attractorPositions = []
    this.syncAttractorUniforms()
  }

  resetCamera() {
    if (!this.camera || !this.controls) return
    this.camera.position.set(0, 35, 45)
    this.controls.target.set(0, 0, 0)
    this.controls.update()
  }

  /* ---- lifecycle ---- */

  constructor(private container: HTMLDivElement) {
    this.init()
    this.createGalaxy()
    this.animate()
  }

  /* ---- initialisation ---- */

  private init() {
    const { clientWidth: w, clientHeight: h } = this.container

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false })
    this.renderer.setSize(w, h)
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1.0
    this.container.appendChild(this.renderer.domElement)

    // Scene
    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color(0x050510)
    this.scene.fog = new THREE.FogExp2(0x050510, 0.006)

    // Camera -- looking down at galaxy from an angle
    this.camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 500)
    this.camera.position.set(0, 35, 45)

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.05
    this.controls.autoRotate = true
    this.controls.autoRotateSpeed = 0.3
    this.controls.minDistance = 10
    this.controls.maxDistance = 150
    this.controls.maxPolarAngle = Math.PI / 2.05

    // Post-processing
    this.composer = new EffectComposer(this.renderer)
    this.composer.addPass(new RenderPass(this.scene, this.camera))
    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(w, h),
      1.5,   // strength
      0.6,   // radius
      0.15   // threshold
    )
    this.composer.addPass(this.bloomPass)

    // Invisible plane for raycast (galaxy disc)
    const planeGeo = new THREE.PlaneGeometry(200, 200)
    const planeMat = new THREE.MeshBasicMaterial({ visible: false })
    this.galaxyPlane = new THREE.Mesh(planeGeo, planeMat)
    this.galaxyPlane.rotation.x = -Math.PI / 2
    this.scene.add(this.galaxyPlane)

    window.addEventListener('resize', this.onResize)
  }

  /* ---- galaxy particle generation ---- */

  private createGalaxy() {
    const positions = new Float32Array(PARTICLE_COUNT * 3)
    const velocities = new Float32Array(PARTICLE_COUNT * 3)
    const masses = new Float32Array(PARTICLE_COUNT)
    const orbitSpeeds = new Float32Array(PARTICLE_COUNT)
    const phases = new Float32Array(PARTICLE_COUNT)
    const radii = new Float32Array(PARTICLE_COUNT)

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // Pick a spiral arm
      const arm = Math.floor(Math.random() * NUM_ARMS)
      const armAngle = (arm / NUM_ARMS) * Math.PI * 2

      // Radius: weighted toward centre (exponential distribution)
      const t = Math.random()
      const r = CORE_RADIUS + (GALAXY_RADIUS - CORE_RADIUS) * Math.pow(t, 0.6)

      // Logarithmic spiral: angle increases with log(r)
      const spiralAngle = armAngle + Math.log(1 + r) * 1.2

      // Gaussian scatter around arm
      const scatter = this.gaussianRandom() * ARM_SPREAD * (r / GALAXY_RADIUS + 0.3)
      const angle = spiralAngle + scatter

      // Vertical scatter (thin disc, thicker in core)
      const heightScale = GALAXY_THICKNESS * (1.0 - 0.6 * (r / GALAXY_RADIUS))
      const y = this.gaussianRandom() * heightScale * 0.5

      // Position
      positions[i * 3 + 0] = Math.cos(angle) * r
      positions[i * 3 + 1] = y
      positions[i * 3 + 2] = Math.sin(angle) * r

      // Velocity (for wobble and colour, not true physics)
      velocities[i * 3 + 0] = (Math.random() - 0.5) * 0.6
      velocities[i * 3 + 1] = y // store vertical offset
      velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.6

      // Mass (affects render size)
      masses[i] = 0.3 + Math.random() * 0.7
      // Bright core stars
      if (r < CORE_RADIUS * 1.5) {
        masses[i] += 0.3
      }

      // Orbit: Keplerian-ish -- inner stars orbit faster
      orbitSpeeds[i] = BASE_ORBIT_SPEED / Math.sqrt(r * 0.5 + 1.0)

      // Phase: the spiral angle is the starting phase
      phases[i] = angle

      // Radius
      radii[i] = r
    }

    // Add some central bulge particles
    const bulgeCount = Math.floor(PARTICLE_COUNT * 0.05)
    for (let i = 0; i < bulgeCount; i++) {
      const idx = i // overwrite first bulgeCount particles
      const r = Math.random() * CORE_RADIUS
      const angle = Math.random() * Math.PI * 2
      const y = this.gaussianRandom() * GALAXY_THICKNESS * 0.8

      positions[idx * 3 + 0] = Math.cos(angle) * r
      positions[idx * 3 + 1] = y
      positions[idx * 3 + 2] = Math.sin(angle) * r

      velocities[idx * 3 + 1] = y
      masses[idx] = 0.5 + Math.random() * 0.8
      orbitSpeeds[idx] = BASE_ORBIT_SPEED * (0.5 + Math.random() * 1.5) / Math.sqrt(r + 1.0)
      phases[idx] = angle
      radii[idx] = r
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('aVelocity', new THREE.BufferAttribute(velocities, 3))
    geo.setAttribute('aMass', new THREE.BufferAttribute(masses, 1))
    geo.setAttribute('aOrbitSpeed', new THREE.BufferAttribute(orbitSpeeds, 1))
    geo.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1))
    geo.setAttribute('aRadius', new THREE.BufferAttribute(radii, 1))

    // Build the initial attractor uniform array (all zeros)
    const attractorArray: THREE.Vector3[] = []
    for (let i = 0; i < MAX_ATTRACTORS; i++) {
      attractorArray.push(new THREE.Vector3(0, 0, 0))
    }

    this.material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0 },
        uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
        uAttractors: { value: attractorArray },
        uAttractorCount: { value: 0 },
        uAttractorStrength: { value: ATTRACTOR_STRENGTH },
        uAttractorSoftening: { value: ATTRACTOR_SOFTENING },
      },
      vertexShader,
      fragmentShader,
      defines: {
        GALAXY_RADIUS: GALAXY_RADIUS.toFixed(1),
      },
    })

    this.points = new THREE.Points(geo, this.material)
    this.scene.add(this.points)
  }

  private gaussianRandom(): number {
    // Box-Muller transform
    let u = 0
    let v = 0
    while (u === 0) u = Math.random()
    while (v === 0) v = Math.random()
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
  }

  private syncAttractorUniforms() {
    if (!this.material) return
    const arr = this.material.uniforms.uAttractors.value as THREE.Vector3[]
    for (let i = 0; i < MAX_ATTRACTORS; i++) {
      if (i < this.attractorPositions.length) {
        arr[i].copy(this.attractorPositions[i])
      } else {
        arr[i].set(0, 0, 0)
      }
    }
    this.material.uniforms.uAttractorCount.value = this.attractorPositions.length
  }

  /* ---- animation loop ---- */

  private animate = () => {
    if (this.disposed) return
    this.animationId = requestAnimationFrame(this.animate)

    const elapsed = this.clock.getElapsedTime()
    this.controls.update()

    if (this.material) {
      this.material.uniforms.uTime.value = elapsed
    }

    this.composer.render()
  }

  /* ---- resize ---- */

  private onResize = () => {
    if (this.disposed) return
    const { clientWidth: w, clientHeight: h } = this.container
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h)
    this.composer.setSize(w, h)
  }

  /* ---- cleanup ---- */

  dispose() {
    this.disposed = true
    cancelAnimationFrame(this.animationId)
    window.removeEventListener('resize', this.onResize)
    this.controls.dispose()

    if (this.points) {
      this.points.geometry.dispose()
      ;(this.points.material as THREE.Material).dispose()
    }

    this.scene.traverse(child => {
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose()
        if (Array.isArray(child.material)) {
          child.material.forEach(m => m.dispose())
        } else {
          child.material.dispose()
        }
      }
    })

    this.renderer.dispose()
    this.composer.dispose()
    if (this.renderer.domElement.parentElement) {
      this.renderer.domElement.parentElement.removeChild(this.renderer.domElement)
    }
  }
}
