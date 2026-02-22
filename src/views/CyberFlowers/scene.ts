import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'

export type FlowerStyle = 'neon' | 'holographic' | 'retro'

interface FlowerData {
  group: THREE.Group
  style: FlowerStyle
  birthTime: number
  swayOffset: number
  swaySpeed: number
  targetScale: number
}

export class CyberFlowerScene {
  private renderer!: THREE.WebGLRenderer
  private scene!: THREE.Scene
  private camera!: THREE.PerspectiveCamera
  private controls!: OrbitControls
  private composer!: EffectComposer
  private bloomPass!: UnrealBloomPass
  private raycaster = new THREE.Raycaster()
  private groundPlane!: THREE.Mesh
  private flowers: FlowerData[] = []
  private particles!: THREE.Points
  private clock = new THREE.Clock()
  private animationId = 0
  private disposed = false

  flowerStyle: FlowerStyle = 'neon'
  accentColor = '#00ffff'

  get autoRotate(): boolean {
    return this.controls?.autoRotate ?? true
  }

  set autoRotate(value: boolean) {
    if (this.controls) this.controls.autoRotate = value
  }

  get autoRotateSpeed(): number {
    return this.controls?.autoRotateSpeed ?? 0.5
  }

  set autoRotateSpeed(value: number) {
    if (this.controls) this.controls.autoRotateSpeed = value
  }

  get bloomStrength(): number {
    return this.bloomPass?.strength ?? 1.0
  }

  set bloomStrength(value: number) {
    if (this.bloomPass) this.bloomPass.strength = value
  }

  get particlesVisible(): boolean {
    return this.particles?.visible ?? true
  }

  set particlesVisible(value: boolean) {
    if (this.particles) this.particles.visible = value
  }

  get flowerCount(): number {
    return this.flowers.length
  }

  resetCamera() {
    if (!this.controls || !this.camera) return
    this.camera.position.set(0, 6, 10)
    this.controls.target.set(0, 1, 0)
    this.controls.update()
  }

  constructor(private container: HTMLDivElement) {
    this.init()
    this.createEnvironment()
    this.createParticles()
    this.animate()
  }

  private init() {
    const { clientWidth: w, clientHeight: h } = this.container

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    this.renderer.setSize(w, h)
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1.2
    this.container.appendChild(this.renderer.domElement)

    // Scene
    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color(0x0a0a0a)
    this.scene.fog = new THREE.FogExp2(0x0a0a0a, 0.035)

    // Camera
    this.camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 100)
    this.camera.position.set(0, 6, 10)

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.05
    this.controls.autoRotate = true
    this.controls.autoRotateSpeed = 0.5
    this.controls.maxPolarAngle = Math.PI / 2.1
    this.controls.minDistance = 3
    this.controls.maxDistance = 25
    this.controls.target.set(0, 1, 0)

    // Post-processing
    this.composer = new EffectComposer(this.renderer)
    this.composer.addPass(new RenderPass(this.scene, this.camera))
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(w, h), 1.0, 0.4, 0.2)
    this.composer.addPass(this.bloomPass)

    // Lights
    const ambient = new THREE.AmbientLight(0x111122, 0.5)
    this.scene.add(ambient)

    const pointLight1 = new THREE.PointLight(0x00ffff, 2, 30)
    pointLight1.position.set(5, 8, 5)
    this.scene.add(pointLight1)

    const pointLight2 = new THREE.PointLight(0xff00ff, 2, 30)
    pointLight2.position.set(-5, 8, -5)
    this.scene.add(pointLight2)

    const pointLight3 = new THREE.PointLight(0xff6ec7, 1.5, 20)
    pointLight3.position.set(0, 3, 8)
    this.scene.add(pointLight3)

    // Resize
    window.addEventListener('resize', this.onResize)
  }

  private createEnvironment() {
    // Grid floor
    const gridSize = 40
    const gridGeo = new THREE.PlaneGeometry(gridSize, gridSize)
    const gridMat = new THREE.ShaderMaterial({
      transparent: true,
      uniforms: {
        uTime: { value: 0 },
        uColor1: { value: new THREE.Color(0x00ffff) },
        uColor2: { value: new THREE.Color(0xff00ff) }
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uTime;
        uniform vec3 uColor1;
        uniform vec3 uColor2;
        varying vec2 vUv;

        void main() {
          vec2 grid = abs(fract(vUv * 20.0 - 0.5) - 0.5);
          float line = min(grid.x, grid.y);
          float gridLine = 1.0 - smoothstep(0.0, 0.03, line);

          vec2 grid2 = abs(fract(vUv * 4.0 - 0.5) - 0.5);
          float line2 = min(grid2.x, grid2.y);
          float majorLine = 1.0 - smoothstep(0.0, 0.02, line2);

          float dist = length(vUv - 0.5) * 2.0;
          float fade = 1.0 - smoothstep(0.2, 1.0, dist);

          vec3 color = mix(uColor1, uColor2, sin(uTime * 0.3 + dist * 3.0) * 0.5 + 0.5);
          float alpha = (gridLine * 0.15 + majorLine * 0.3) * fade;

          gl_FragColor = vec4(color, alpha);
        }
      `
    })
    const grid = new THREE.Mesh(gridGeo, gridMat)
    grid.rotation.x = -Math.PI / 2
    grid.position.y = -0.01
    this.scene.add(grid)

    // Invisible ground for raycasting
    const groundGeo = new THREE.PlaneGeometry(gridSize, gridSize)
    const groundMat = new THREE.MeshBasicMaterial({ visible: false })
    this.groundPlane = new THREE.Mesh(groundGeo, groundMat)
    this.groundPlane.rotation.x = -Math.PI / 2
    this.scene.add(this.groundPlane)
  }

  private createParticles() {
    const count = 500
    const positions = new Float32Array(count * 3)
    const colors = new Float32Array(count * 3)
    const sizes = new Float32Array(count)

    const palette = [
      new THREE.Color(0x00ffff),
      new THREE.Color(0xff00ff),
      new THREE.Color(0xff6ec7),
      new THREE.Color(0x7b68ee)
    ]

    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 30
      positions[i * 3 + 1] = Math.random() * 15
      positions[i * 3 + 2] = (Math.random() - 0.5) * 30

      const c = palette[Math.floor(Math.random() * palette.length)]
      colors[i * 3] = c.r
      colors[i * 3 + 1] = c.g
      colors[i * 3 + 2] = c.b

      sizes[i] = Math.random() * 3 + 1
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1))

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      vertexColors: true,
      uniforms: {
        uTime: { value: 0 },
        uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) }
      },
      vertexShader: `
        attribute float size;
        varying vec3 vColor;
        uniform float uTime;
        uniform float uPixelRatio;

        void main() {
          vColor = color;
          vec3 pos = position;
          pos.y += sin(uTime * 0.5 + position.x * 0.5) * 0.3;
          pos.x += cos(uTime * 0.3 + position.z * 0.3) * 0.2;

          vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
          gl_Position = projectionMatrix * mvPos;
          gl_PointSize = size * uPixelRatio * (8.0 / -mvPos.z);
        }
      `,
      fragmentShader: `
        varying vec3 vColor;

        void main() {
          float d = length(gl_PointCoord - 0.5);
          if (d > 0.5) discard;
          float alpha = 1.0 - smoothstep(0.2, 0.5, d);
          gl_FragColor = vec4(vColor, alpha * 0.6);
        }
      `
    })

    this.particles = new THREE.Points(geo, mat)
    this.scene.add(this.particles)
  }

  plantFlower(ndcX: number, ndcY: number) {
    const mouse = new THREE.Vector2(ndcX, ndcY)
    this.raycaster.setFromCamera(mouse, this.camera)
    const hits = this.raycaster.intersectObject(this.groundPlane)
    if (hits.length === 0) return

    const point = hits[0].point
    // Clamp to a reasonable area
    if (Math.abs(point.x) > 12 || Math.abs(point.z) > 12) return

    const flower = this.createFlower(this.flowerStyle, new THREE.Color(this.accentColor))
    flower.position.copy(point)
    flower.position.y = 0
    this.scene.add(flower)

    this.flowers.push({
      group: flower,
      style: this.flowerStyle,
      birthTime: this.clock.getElapsedTime(),
      swayOffset: Math.random() * Math.PI * 2,
      swaySpeed: 0.5 + Math.random() * 0.5,
      targetScale: 0.8 + Math.random() * 0.4
    })
  }

  private createFlower(style: FlowerStyle, accent: THREE.Color): THREE.Group {
    const group = new THREE.Group()
    group.scale.set(0.01, 0.01, 0.01) // Start tiny for grow animation

    switch (style) {
      case 'neon':
        this.buildNeonFlower(group, accent)
        break
      case 'holographic':
        this.buildHolographicFlower(group, accent)
        break
      case 'retro':
        this.buildRetroFlower(group, accent)
        break
    }

    return group
  }

  private buildNeonFlower(group: THREE.Group, accent: THREE.Color) {
    const stemHeight = 1.5 + Math.random() * 1.5
    const petalCount = 5 + Math.floor(Math.random() * 4)

    // Stem - glowing tube
    const stemCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0.1, stemHeight * 0.3, 0.05),
      new THREE.Vector3(-0.05, stemHeight * 0.7, -0.05),
      new THREE.Vector3(0, stemHeight, 0)
    ])
    const stemGeo = new THREE.TubeGeometry(stemCurve, 16, 0.04, 8, false)
    const stemMat = new THREE.MeshStandardMaterial({
      color: accent,
      emissive: accent,
      emissiveIntensity: 0.8,
      roughness: 0.3
    })
    group.add(new THREE.Mesh(stemGeo, stemMat))

    // Wireframe accent on stem
    const wireGeo = new THREE.TubeGeometry(stemCurve, 8, 0.06, 6, false)
    const wireMat = new THREE.MeshBasicMaterial({
      color: accent,
      wireframe: true,
      transparent: true,
      opacity: 0.3
    })
    group.add(new THREE.Mesh(wireGeo, wireMat))

    // Center sphere
    const centerGeo = new THREE.SphereGeometry(0.15, 16, 16)
    const centerMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: accent,
      emissiveIntensity: 2.0
    })
    const center = new THREE.Mesh(centerGeo, centerMat)
    center.position.y = stemHeight
    group.add(center)

    // Petals
    for (let i = 0; i < petalCount; i++) {
      const angle = (i / petalCount) * Math.PI * 2
      const petalShape = new THREE.Shape()
      petalShape.moveTo(0, 0)
      petalShape.bezierCurveTo(0.15, 0.1, 0.2, 0.4, 0, 0.6)
      petalShape.bezierCurveTo(-0.2, 0.4, -0.15, 0.1, 0, 0)

      const petalGeo = new THREE.ShapeGeometry(petalShape)
      const petalMat = new THREE.MeshStandardMaterial({
        color: accent,
        emissive: accent,
        emissiveIntensity: 1.2,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.85
      })
      const petal = new THREE.Mesh(petalGeo, petalMat)
      petal.position.y = stemHeight
      petal.rotation.y = angle
      petal.rotation.x = -0.3 - Math.random() * 0.3
      group.add(petal)
    }
  }

  private buildHolographicFlower(group: THREE.Group, accent: THREE.Color) {
    const stemHeight = 1.5 + Math.random() * 1.5
    const petalCount = 6 + Math.floor(Math.random() * 3)

    // Stem - iridescent
    const stemCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(-0.08, stemHeight * 0.4, 0.08),
      new THREE.Vector3(0.05, stemHeight * 0.6, -0.03),
      new THREE.Vector3(0, stemHeight, 0)
    ])
    const stemGeo = new THREE.TubeGeometry(stemCurve, 16, 0.035, 8, false)
    const stemMat = new THREE.ShaderMaterial({
      transparent: true,
      side: THREE.DoubleSide,
      uniforms: {
        uTime: { value: 0 },
        uAccent: { value: accent }
      },
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vViewDir;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          vViewDir = normalize(-mvPos.xyz);
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: `
        uniform float uTime;
        uniform vec3 uAccent;
        varying vec3 vNormal;
        varying vec3 vViewDir;

        void main() {
          float fresnel = pow(1.0 - abs(dot(vNormal, vViewDir)), 3.0);
          vec3 col = mix(uAccent, vec3(1.0, 0.4, 1.0), fresnel);
          col += vec3(0.2, 0.5, 1.0) * fresnel * 0.5;
          gl_FragColor = vec4(col, 0.6 + fresnel * 0.4);
        }
      `
    })
    group.add(new THREE.Mesh(stemGeo, stemMat))

    // Center - glowing orb
    const centerGeo = new THREE.IcosahedronGeometry(0.18, 2)
    const centerMat = new THREE.ShaderMaterial({
      transparent: true,
      uniforms: {
        uTime: { value: 0 },
        uAccent: { value: accent }
      },
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vViewDir;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          vViewDir = normalize(-mvPos.xyz);
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: `
        uniform float uTime;
        uniform vec3 uAccent;
        varying vec3 vNormal;
        varying vec3 vViewDir;
        void main() {
          float fresnel = pow(1.0 - abs(dot(vNormal, vViewDir)), 2.0);
          vec3 rainbow = 0.5 + 0.5 * cos(6.28318 * (fresnel + uTime * 0.2 + vec3(0.0, 0.33, 0.67)));
          vec3 col = mix(uAccent, rainbow, 0.7);
          gl_FragColor = vec4(col, 0.7 + fresnel * 0.3);
        }
      `
    })
    const center = new THREE.Mesh(centerGeo, centerMat)
    center.position.y = stemHeight
    group.add(center)

    // Petals - translucent iridescent
    for (let i = 0; i < petalCount; i++) {
      const angle = (i / petalCount) * Math.PI * 2
      const petalGeo = new THREE.CircleGeometry(0.35, 16, 0, Math.PI)
      const petalMat = new THREE.ShaderMaterial({
        transparent: true,
        side: THREE.DoubleSide,
        uniforms: {
          uTime: { value: 0 },
          uAccent: { value: accent },
          uIndex: { value: i / petalCount }
        },
        vertexShader: `
          varying vec2 vUv;
          varying vec3 vNormal;
          varying vec3 vViewDir;
          void main() {
            vUv = uv;
            vNormal = normalize(normalMatrix * normal);
            vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
            vViewDir = normalize(-mvPos.xyz);
            gl_Position = projectionMatrix * mvPos;
          }
        `,
        fragmentShader: `
          uniform float uTime;
          uniform vec3 uAccent;
          uniform float uIndex;
          varying vec2 vUv;
          varying vec3 vNormal;
          varying vec3 vViewDir;
          void main() {
            float fresnel = pow(1.0 - abs(dot(vNormal, vViewDir)), 2.5);
            vec3 rainbow = 0.5 + 0.5 * cos(6.28318 * (fresnel * 2.0 + uTime * 0.15 + uIndex + vec3(0.0, 0.33, 0.67)));
            vec3 col = mix(uAccent * 0.5, rainbow, fresnel * 0.8 + 0.2);
            float alpha = (0.3 + fresnel * 0.5) * smoothstep(0.0, 0.1, vUv.y);
            gl_FragColor = vec4(col, alpha);
          }
        `
      })
      const petal = new THREE.Mesh(petalGeo, petalMat)
      petal.position.y = stemHeight
      petal.rotation.y = angle
      petal.rotation.x = -0.5 - Math.random() * 0.2
      group.add(petal)
    }
  }

  private buildRetroFlower(group: THREE.Group, accent: THREE.Color) {
    const stemHeight = 1.2 + Math.random() * 1.8
    const petalCount = 4 + Math.floor(Math.random() * 4)

    // Stem - chrome cylinder
    const stemCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0.12, stemHeight * 0.5, 0.06),
      new THREE.Vector3(-0.06, stemHeight * 0.8, -0.04),
      new THREE.Vector3(0, stemHeight, 0)
    ])
    const stemGeo = new THREE.TubeGeometry(stemCurve, 12, 0.045, 6, false)
    const stemMat = new THREE.MeshPhysicalMaterial({
      color: 0xaaaaaa,
      metalness: 0.9,
      roughness: 0.1,
      envMapIntensity: 1.5,
      clearcoat: 1.0
    })
    group.add(new THREE.Mesh(stemGeo, stemMat))

    // Center - geometric dodecahedron
    const centerGeo = new THREE.DodecahedronGeometry(0.2, 0)
    const centerMat = new THREE.MeshPhysicalMaterial({
      color: accent,
      metalness: 0.8,
      roughness: 0.15,
      emissive: accent,
      emissiveIntensity: 0.6,
      clearcoat: 1.0
    })
    const center = new THREE.Mesh(centerGeo, centerMat)
    center.position.y = stemHeight
    group.add(center)

    // Petals - geometric triangles / diamonds
    const retroPalette = [0xff6ec7, 0x7b68ee, 0x00ced1, 0xff1493]
    for (let i = 0; i < petalCount; i++) {
      const angle = (i / petalCount) * Math.PI * 2
      // Diamond shape
      const shape = new THREE.Shape()
      shape.moveTo(0, 0)
      shape.lineTo(0.15, 0.25)
      shape.lineTo(0, 0.55)
      shape.lineTo(-0.15, 0.25)
      shape.closePath()

      const petalGeo = new THREE.ShapeGeometry(shape)
      const petalColor = new THREE.Color(retroPalette[i % retroPalette.length])
      const petalMat = new THREE.MeshPhysicalMaterial({
        color: petalColor,
        metalness: 0.7,
        roughness: 0.2,
        emissive: petalColor,
        emissiveIntensity: 0.4,
        side: THREE.DoubleSide,
        clearcoat: 0.8
      })
      const petal = new THREE.Mesh(petalGeo, petalMat)
      petal.position.y = stemHeight
      petal.rotation.y = angle
      petal.rotation.x = -0.4 - Math.random() * 0.3

      // Add wireframe overlay for retro feel
      const wireGeo = new THREE.EdgesGeometry(petalGeo)
      const wireMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.4 })
      petal.add(new THREE.LineSegments(wireGeo, wireMat))

      group.add(petal)
    }
  }

  clearFlowers() {
    for (const f of this.flowers) {
      this.scene.remove(f.group)
      f.group.traverse(child => {
        if (child instanceof THREE.Mesh) {
          child.geometry.dispose()
          if (Array.isArray(child.material)) {
            child.material.forEach(m => m.dispose())
          } else {
            child.material.dispose()
          }
        }
      })
    }
    this.flowers = []
  }

  private animate = () => {
    if (this.disposed) return
    this.animationId = requestAnimationFrame(this.animate)

    const elapsed = this.clock.getElapsedTime()

    // Update controls
    this.controls.update()

    // Update grid shader time
    this.scene.traverse(child => {
      if (child instanceof THREE.Mesh && child.material instanceof THREE.ShaderMaterial) {
        if (child.material.uniforms.uTime) {
          child.material.uniforms.uTime.value = elapsed
        }
      }
    })

    // Update particle shader
    if (this.particles) {
      const mat = this.particles.material as THREE.ShaderMaterial
      mat.uniforms.uTime.value = elapsed
    }

    // Animate flowers
    for (const f of this.flowers) {
      const age = elapsed - f.birthTime
      // Grow animation
      const growProgress = Math.min(age / 0.8, 1)
      const eased = 1 - Math.pow(1 - growProgress, 3) // ease-out cubic
      const scale = eased * f.targetScale
      f.group.scale.set(scale, scale, scale)

      // Gentle sway
      const sway = Math.sin(elapsed * f.swaySpeed + f.swayOffset) * 0.03
      f.group.rotation.x = sway
      f.group.rotation.z = sway * 0.7
    }

    this.composer.render()
  }

  private onResize = () => {
    if (this.disposed) return
    const { clientWidth: w, clientHeight: h } = this.container
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h)
    this.composer.setSize(w, h)
  }

  dispose() {
    this.disposed = true
    cancelAnimationFrame(this.animationId)
    window.removeEventListener('resize', this.onResize)
    this.controls.dispose()
    this.clearFlowers()
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
