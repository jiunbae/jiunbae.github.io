import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'

export type VisualizationMode = 'terrain' | 'radial' | 'waveform'

export class AudioVisualizerScene {
  private renderer!: THREE.WebGLRenderer
  private scene!: THREE.Scene
  private camera!: THREE.PerspectiveCamera
  private controls!: OrbitControls
  private composer!: EffectComposer
  private bloomPass!: UnrealBloomPass
  private userBloomStrength = 1.5
  private clock = new THREE.Clock()
  private animationId = 0
  private disposed = false

  // Audio
  private audioContext: AudioContext | null = null
  private analyser: AnalyserNode | null = null
  private sourceNode: AudioBufferSourceNode | MediaStreamAudioSourceNode | null = null
  private mediaStream: MediaStream | null = null
  private frequencyData: Uint8Array = new Uint8Array(0)
  private timeDomainData: Uint8Array = new Uint8Array(0)
  private isPlaying = false

  // Visualization objects
  private currentMode: VisualizationMode = 'terrain'
  private terrainMesh: THREE.Mesh | null = null
  private radialGroup: THREE.Group | null = null
  private radialBars: THREE.Mesh[] = []
  private waveformMesh: THREE.Mesh | null = null

  // Camera presets per mode
  private cameraTargets: Record<VisualizationMode, { pos: THREE.Vector3; target: THREE.Vector3 }> = {
    terrain: { pos: new THREE.Vector3(0, 12, 18), target: new THREE.Vector3(0, 0, 0) },
    radial: { pos: new THREE.Vector3(0, 0, 20), target: new THREE.Vector3(0, 0, 0) },
    waveform: { pos: new THREE.Vector3(0, 5, 16), target: new THREE.Vector3(0, 0, 0) }
  }

  // Public properties
  private _autoRotate = true

  get autoRotate(): boolean {
    return this._autoRotate
  }

  set autoRotate(value: boolean) {
    this._autoRotate = value
    if (this.controls) this.controls.autoRotate = value
  }

  get bloomStrength(): number {
    return this.bloomPass?.strength ?? 1.5
  }

  set bloomStrength(value: number) {
    this.userBloomStrength = value
    if (this.bloomPass) this.bloomPass.strength = value
  }

  constructor(private container: HTMLDivElement) {
    this.init()
    this.buildTerrain()
    this.buildRadial()
    this.buildWaveform()
    this.setMode('terrain')
    this.animate()
  }

  private init() {
    const { clientWidth: w, clientHeight: h } = this.container

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    this.renderer.setSize(w, h)
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1.0
    this.container.appendChild(this.renderer.domElement)

    this.renderer.domElement.addEventListener('webglcontextlost', this.handleContextLost)
    this.renderer.domElement.addEventListener('webglcontextrestored', this.handleContextRestored)

    // Scene
    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color(0x080810)
    this.scene.fog = new THREE.FogExp2(0x080810, 0.012)

    // Camera
    this.camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 200)
    this.camera.position.set(0, 12, 18)

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.05
    this.controls.autoRotate = this._autoRotate
    this.controls.autoRotateSpeed = 0.3
    this.controls.minDistance = 5
    this.controls.maxDistance = 60

    // Post-processing
    this.composer = new EffectComposer(this.renderer)
    this.composer.addPass(new RenderPass(this.scene, this.camera))
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(w, h), 1.5, 0.6, 0.15)
    this.composer.addPass(this.bloomPass)

    // Lights
    const ambient = new THREE.AmbientLight(0x222244, 0.6)
    this.scene.add(ambient)

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.4)
    dirLight.position.set(10, 20, 10)
    this.scene.add(dirLight)

    // Resize
    window.addEventListener('resize', this.onResize)
  }

  // --- Audio setup ---

  private ensureAudioContext() {
    if (this.disposed) return
    if (!this.audioContext || this.audioContext.state === 'closed') {
      this.audioContext = new AudioContext()
    }
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume()
    }
    return this.audioContext
  }

  private setupAnalyser(source: AudioNode) {
    const ctx = this.ensureAudioContext()
    if (!ctx) return
    this.analyser = ctx.createAnalyser()
    this.analyser.fftSize = 2048
    this.analyser.smoothingTimeConstant = 0.8
    source.connect(this.analyser)
    this.analyser.connect(ctx.destination)
    this.frequencyData = new Uint8Array(this.analyser.frequencyBinCount)
    this.timeDomainData = new Uint8Array(this.analyser.fftSize)
    this.isPlaying = true
  }

  async loadAudioFile(file: File) {
    this.stopAudio()
    const ctx = this.ensureAudioContext()
    if (!ctx) return
    const arrayBuffer = await file.arrayBuffer()
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer)

    const source = ctx.createBufferSource()
    source.buffer = audioBuffer
    source.loop = true
    this.sourceNode = source
    this.setupAnalyser(source)
    // Disconnect analyser from destination for file playback and reconnect properly
    // analyser is already connected in setupAnalyser
    source.start(0)
  }

  async startMic() {
    this.stopAudio()
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      this.mediaStream = stream
      const ctx = this.ensureAudioContext()
      if (!ctx) return
      const source = ctx.createMediaStreamSource(stream)
      this.sourceNode = source

      // For mic, don't connect analyser to destination to avoid feedback
      this.analyser = ctx.createAnalyser()
      this.analyser.fftSize = 2048
      this.analyser.smoothingTimeConstant = 0.8
      source.connect(this.analyser)
      this.frequencyData = new Uint8Array(this.analyser.frequencyBinCount)
      this.timeDomainData = new Uint8Array(this.analyser.fftSize)
      this.isPlaying = true
    } catch (err) {
      console.error('Microphone access denied:', err)
      throw err
    }
  }

  stopAudio() {
    this.isPlaying = false
    if (this.sourceNode) {
      try {
        this.sourceNode.disconnect()
        if (this.sourceNode instanceof AudioBufferSourceNode) {
          this.sourceNode.stop()
        }
      } catch {
        // already stopped
      }
      this.sourceNode = null
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(t => t.stop())
      this.mediaStream = null
    }
    if (this.analyser) {
      try {
        this.analyser.disconnect()
      } catch {
        // already disconnected
      }
      this.analyser = null
    }
    this.frequencyData = new Uint8Array(0)
    this.timeDomainData = new Uint8Array(0)
  }

  // --- Visualization modes ---

  setMode(mode: VisualizationMode) {
    this.currentMode = mode

    // Toggle visibility
    if (this.terrainMesh) this.terrainMesh.visible = mode === 'terrain'
    if (this.radialGroup) this.radialGroup.visible = mode === 'radial'
    if (this.waveformMesh) this.waveformMesh.visible = mode === 'waveform'

    // Animate camera to preset
    const preset = this.cameraTargets[mode]
    this.camera.position.copy(preset.pos)
    this.controls.target.copy(preset.target)
    this.controls.update()
  }

  // --- Build visualizations ---

  private buildTerrain() {
    const segments = 128
    const size = 40
    const geometry = new THREE.PlaneGeometry(size, size, segments - 1, segments - 1)
    geometry.rotateX(-Math.PI / 2)

    const material = new THREE.ShaderMaterial({
      vertexColors: true,
      wireframe: true,
      transparent: true,
      uniforms: {
        uOpacity: { value: 0.85 }
      },
      vertexShader: `
        attribute vec3 customColor;
        varying vec3 vColor;
        void main() {
          vColor = customColor;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uOpacity;
        varying vec3 vColor;
        void main() {
          gl_FragColor = vec4(vColor, uOpacity);
        }
      `
    })

    // Initialize color attribute
    const count = geometry.attributes.position.count
    const colors = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      colors[i * 3] = 0.0
      colors[i * 3 + 1] = 0.1
      colors[i * 3 + 2] = 0.4
    }
    geometry.setAttribute('customColor', new THREE.BufferAttribute(colors, 3))

    this.terrainMesh = new THREE.Mesh(geometry, material)
    this.scene.add(this.terrainMesh)
  }

  private buildRadial() {
    this.radialGroup = new THREE.Group()
    this.radialBars = []
    const barCount = 128
    const radius = 8

    for (let i = 0; i < barCount; i++) {
      const angle = (i / barCount) * Math.PI * 2
      const geometry = new THREE.BoxGeometry(0.25, 1, 0.25)
      // Shift geometry pivot to bottom
      geometry.translate(0, 0.5, 0)

      const hue = i / barCount
      const color = new THREE.Color().setHSL(hue, 1.0, 0.5)
      const material = new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.6,
        roughness: 0.3,
        metalness: 0.5
      })

      const bar = new THREE.Mesh(geometry, material)
      bar.position.x = Math.cos(angle) * radius
      bar.position.z = Math.sin(angle) * radius
      bar.lookAt(0, 0, 0)
      bar.rotateX(Math.PI / 2)

      this.radialBars.push(bar)
      this.radialGroup.add(bar)
    }

    this.scene.add(this.radialGroup)
  }

  private buildWaveform() {
    // Create a flat ribbon that will be deformed by waveform data
    const lengthSegments = 256
    const widthSegments = 4
    const geometry = new THREE.PlaneGeometry(30, 3, lengthSegments, widthSegments)

    const material = new THREE.ShaderMaterial({
      side: THREE.DoubleSide,
      transparent: true,
      uniforms: {
        uTime: { value: 0 },
        uIntensity: { value: 1.0 }
      },
      vertexShader: `
        varying vec2 vUv;
        varying float vDisplacement;
        void main() {
          vUv = uv;
          vDisplacement = position.y;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uTime;
        uniform float uIntensity;
        varying vec2 vUv;
        varying float vDisplacement;
        void main() {
          float d = abs(vDisplacement) * 0.5;
          vec3 col1 = vec3(0.1, 0.3, 1.0);
          vec3 col2 = vec3(1.0, 0.2, 0.8);
          vec3 col3 = vec3(0.0, 1.0, 0.6);
          vec3 color = mix(col1, col2, d);
          color = mix(color, col3, sin(vUv.x * 6.28 + uTime) * 0.3 + 0.3);
          float glow = exp(-d * 2.0) * uIntensity;
          color += glow * 0.4;
          float alpha = 0.6 + glow * 0.4;
          gl_FragColor = vec4(color, alpha);
        }
      `
    })

    this.waveformMesh = new THREE.Mesh(geometry, material)
    this.scene.add(this.waveformMesh)
  }

  // --- Update visualizations ---

  private updateTerrain() {
    if (!this.terrainMesh || this.frequencyData.length === 0) return
    const geometry = this.terrainMesh.geometry
    const positions = geometry.attributes.position
    const colors = geometry.attributes.customColor as THREE.BufferAttribute
    const segments = 128
    const freqBins = this.frequencyData.length

    // Sample frequency data to fill 128 columns
    for (let ix = 0; ix < segments; ix++) {
      const freqIndex = Math.floor((ix / segments) * Math.min(freqBins, 512))
      const amplitude = (this.frequencyData[freqIndex] || 0) / 255

      for (let iz = 0; iz < segments; iz++) {
        const vertexIndex = iz * segments + ix
        // Displace Y
        const dist = Math.abs(iz - segments / 2) / (segments / 2)
        const falloff = 1.0 - dist * dist
        const y = amplitude * 8 * falloff

        positions.setY(vertexIndex, y)

        // Color by height
        const t = amplitude * falloff
        // Low=blue(0,0.3,1) mid=green(0,1,0.3) high=red(1,0.2,0)
        let r: number, g: number, b: number
        if (t < 0.5) {
          const s = t / 0.5
          r = s * 0.0
          g = 0.3 + s * 0.7
          b = 1.0 - s * 0.7
        } else {
          const s = (t - 0.5) / 0.5
          r = s * 1.0
          g = 1.0 - s * 0.8
          b = 0.3 - s * 0.3
        }
        colors.setXYZ(vertexIndex, r, g, b)
      }
    }
    positions.needsUpdate = true
    colors.needsUpdate = true
  }

  private updateRadial() {
    if (!this.radialGroup || this.frequencyData.length === 0) return
    const barCount = this.radialBars.length
    const freqBins = this.frequencyData.length

    for (let i = 0; i < barCount; i++) {
      const freqIndex = Math.floor((i / barCount) * Math.min(freqBins, 512))
      const amplitude = (this.frequencyData[freqIndex] || 0) / 255
      const scaleY = 0.2 + amplitude * 12
      this.radialBars[i].scale.y = scaleY

      // Update emissive intensity based on amplitude
      const mat = this.radialBars[i].material as THREE.MeshStandardMaterial
      mat.emissiveIntensity = 0.3 + amplitude * 1.5
    }
  }

  private updateWaveform() {
    if (!this.waveformMesh || this.timeDomainData.length === 0) return
    const geometry = this.waveformMesh.geometry
    const positions = geometry.attributes.position
    const count = positions.count
    const lengthSegments = 257 // PlaneGeometry(_, _, 256, 4) => 257 cols, 5 rows
    const widthSegments = 5

    for (let ix = 0; ix < lengthSegments; ix++) {
      const dataIndex = Math.floor((ix / lengthSegments) * this.timeDomainData.length)
      const sample = ((this.timeDomainData[dataIndex] || 128) - 128) / 128

      for (let iy = 0; iy < widthSegments; iy++) {
        const vertexIndex = iy * lengthSegments + ix
        if (vertexIndex >= count) continue
        // Center rows get full displacement, edges taper
        const rowT = Math.abs(iy - 2) / 2
        const taper = 1.0 - rowT
        positions.setY(vertexIndex, sample * 4 * taper)

        // Add subtle Z depth undulation
        const z = Math.sin(ix * 0.05 + this.clock.getElapsedTime() * 0.5) * 2 * taper
        positions.setZ(vertexIndex, (iy - 2) * 0.6 + z * 0.3)
      }
    }
    positions.needsUpdate = true
  }

  // --- Animation loop ---

  private handleContextLost = (e: Event) => {
    e.preventDefault()
    cancelAnimationFrame(this.animationId)
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
    this.animationId = requestAnimationFrame(this.animate)

    const elapsed = this.clock.getElapsedTime()

    // Read audio data
    if (this.isPlaying && this.analyser) {
      this.analyser.getByteFrequencyData(this.frequencyData)
      this.analyser.getByteTimeDomainData(this.timeDomainData)
    }

    // Dynamic bloom based on bass
    if (this.isPlaying && this.frequencyData.length > 0) {
      let bassSum = 0
      const bassBins = Math.min(8, this.frequencyData.length)
      for (let i = 0; i < bassBins; i++) {
        bassSum += this.frequencyData[i]
      }
      const bassAvg = bassSum / bassBins / 255
      // Modulate around the user-set strength (use stored value to avoid feedback decay)
      this.bloomPass.strength = this.userBloomStrength * (0.7 + bassAvg * 0.6)
    }

    // Update active visualization
    switch (this.currentMode) {
      case 'terrain':
        this.updateTerrain()
        break
      case 'radial':
        this.updateRadial()
        // Gentle rotation of the radial group
        if (this.radialGroup) {
          this.radialGroup.rotation.y += 0.002
        }
        break
      case 'waveform':
        this.updateWaveform()
        if (this.waveformMesh) {
          const mat = this.waveformMesh.material as THREE.ShaderMaterial
          mat.uniforms.uTime.value = elapsed
        }
        break
    }

    // Gently idle-animate terrain even without audio
    if (this.currentMode === 'terrain' && !this.isPlaying && this.terrainMesh) {
      const geo = this.terrainMesh.geometry
      const pos = geo.attributes.position
      const cols = geo.attributes.customColor as THREE.BufferAttribute
      const segments = 128
      for (let ix = 0; ix < segments; ix++) {
        for (let iz = 0; iz < segments; iz++) {
          const vi = iz * segments + ix
          const y = Math.sin(ix * 0.08 + elapsed * 0.5) * Math.cos(iz * 0.08 + elapsed * 0.3) * 0.5
          pos.setY(vi, y)
          const t = (y + 0.5) / 1.0
          cols.setXYZ(vi, t * 0.2, 0.1 + t * 0.3, 0.4 + (1 - t) * 0.3)
        }
      }
      pos.needsUpdate = true
      cols.needsUpdate = true
    }

    this.controls.update()
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
    this.stopAudio()
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close()
    }
    this.controls.dispose()
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
    this.renderer.domElement.removeEventListener('webglcontextlost', this.handleContextLost)
    this.renderer.domElement.removeEventListener('webglcontextrestored', this.handleContextRestored)
    this.renderer.dispose()
    this.composer.dispose()
    if (this.renderer.domElement.parentElement) {
      this.renderer.domElement.parentElement.removeChild(this.renderer.domElement)
    }
  }
}
