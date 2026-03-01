/**
 * GPU-accelerated Navier-Stokes fluid simulation using WebGL.
 *
 * Pipeline (per frame):
 *   1. Advect velocity (semi-Lagrangian)
 *   2. Diffuse velocity (Jacobi)
 *   3. Add external forces (mouse / auto-splat)
 *   4. Vorticity confinement
 *   5. Compute divergence
 *   6. Pressure solve (Jacobi)
 *   7. Subtract pressure gradient
 *   8. Advect dye
 *   9. Display (bloom-like glow)
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface FluidConfig {
  viscosity: number       // 0 .. 1
  curl: number            // vorticity confinement strength
  pressure: number        // pressure iteration count
  splatRadius: number     // base splat radius in sim-space
}

interface DoubleFBO {
  read: FBOHandle
  write: FBOHandle
  swap(): void
}

interface FBOHandle {
  texture: WebGLTexture
  fbo: WebGLFramebuffer
  width: number
  height: number
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function compileShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)
  if (!shader) throw new Error('Failed to create shader — WebGL context may be lost')
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader)
    gl.deleteShader(shader)
    throw new Error(`Shader compile error: ${info}`)
  }
  return shader
}

function createProgram(gl: WebGLRenderingContext, vertSrc: string, fragSrc: string) {
  const program = gl.createProgram()
  if (!program) throw new Error('Failed to create program — WebGL context may be lost')
  gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, vertSrc))
  gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, fragSrc))
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`Program link error: ${gl.getProgramInfoLog(program)}`)
  }
  return program
}

function getUniforms(gl: WebGLRenderingContext, program: WebGLProgram): Record<string, WebGLUniformLocation> {
  const uniforms: Record<string, WebGLUniformLocation> = {}
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS)
  for (let i = 0; i < count; i++) {
    const info = gl.getActiveUniform(program, i)!
    const loc = gl.getUniformLocation(program, info.name)
    if (loc) uniforms[info.name] = loc
  }
  return uniforms
}

// ---------------------------------------------------------------------------
// Shader sources
// ---------------------------------------------------------------------------

const BASE_VERT = `
  attribute vec2 aPosition;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform vec2 texelSize;
  void main () {
    vUv = aPosition * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(aPosition, 0.0, 1.0);
  }
`

const SPLAT_FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTarget;
  uniform float aspectRatio;
  uniform vec3 color;
  uniform vec2 point;
  uniform float radius;
  void main () {
    vec2 p = vUv - point;
    p.x *= aspectRatio;
    vec3 splat = exp(-dot(p, p) / radius) * color;
    vec3 base = texture2D(uTarget, vUv).xyz;
    gl_FragColor = vec4(base + splat, 1.0);
  }
`

const ADVECTION_FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uVelocity;
  uniform sampler2D uSource;
  uniform vec2 texelSize;
  uniform float dt;
  uniform float dissipation;
  void main () {
    vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
    vec3 result = dissipation * texture2D(uSource, coord).xyz;
    gl_FragColor = vec4(result, 1.0);
  }
`

const DIVERGENCE_FRAG = `
  precision highp float;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uVelocity;
  void main () {
    float L = texture2D(uVelocity, vL).x;
    float R = texture2D(uVelocity, vR).x;
    float T = texture2D(uVelocity, vT).y;
    float B = texture2D(uVelocity, vB).y;
    float div = 0.5 * (R - L + T - B);
    gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
  }
`

const PRESSURE_FRAG = `
  precision highp float;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uPressure;
  uniform sampler2D uDivergence;
  void main () {
    float L = texture2D(uPressure, vL).x;
    float R = texture2D(uPressure, vR).x;
    float T = texture2D(uPressure, vT).x;
    float B = texture2D(uPressure, vB).x;
    float div = texture2D(uDivergence, vUv).x;
    float pressure = (L + R + B + T - div) * 0.25;
    gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
  }
`

const GRADIENT_SUB_FRAG = `
  precision highp float;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uPressure;
  uniform sampler2D uVelocity;
  void main () {
    float L = texture2D(uPressure, vL).x;
    float R = texture2D(uPressure, vR).x;
    float T = texture2D(uPressure, vT).x;
    float B = texture2D(uPressure, vB).x;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity.xy -= vec2(R - L, T - B);
    gl_FragColor = vec4(velocity, 0.0, 1.0);
  }
`

const CURL_FRAG = `
  precision highp float;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uVelocity;
  void main () {
    float L = texture2D(uVelocity, vL).y;
    float R = texture2D(uVelocity, vR).y;
    float T = texture2D(uVelocity, vT).x;
    float B = texture2D(uVelocity, vB).x;
    float vorticity = R - L - T + B;
    gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
  }
`

const VORTICITY_FRAG = `
  precision highp float;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uVelocity;
  uniform sampler2D uCurl;
  uniform float curl;
  uniform float dt;
  void main () {
    float L = texture2D(uCurl, vL).x;
    float R = texture2D(uCurl, vR).x;
    float T = texture2D(uCurl, vT).x;
    float B = texture2D(uCurl, vB).x;
    float C = texture2D(uCurl, vUv).x;
    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
    force /= length(force) + 0.0001;
    force *= curl * C;
    force.y *= -1.0;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity += force * dt;
    gl_FragColor = vec4(velocity, 0.0, 1.0);
  }
`

const DISPLAY_FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTexture;
  void main () {
    vec3 c = texture2D(uTexture, vUv).rgb;
    // Subtle bloom: brighten saturated/bright areas
    float brightness = max(c.r, max(c.g, c.b));
    vec3 bloom = c * smoothstep(0.4, 1.2, brightness) * 0.6;
    c += bloom;
    // Tone-map
    c = c / (1.0 + c);
    // Slight vignette
    vec2 uv = vUv * 2.0 - 1.0;
    float vig = 1.0 - dot(uv * 0.4, uv * 0.4);
    c *= vig;
    gl_FragColor = vec4(c, 1.0);
  }
`

const CLEAR_FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTexture;
  uniform float value;
  void main () {
    gl_FragColor = value * texture2D(uTexture, vUv);
  }
`

// ---------------------------------------------------------------------------
// Context loss overlay
// ---------------------------------------------------------------------------

function showContextLostOverlay(container: HTMLElement): HTMLDivElement {
  const overlay = document.createElement('div')
  overlay.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.8);color:#fff;font:14px/1.5 sans-serif;cursor:pointer;z-index:100'
  overlay.textContent = 'WebGL context lost — click to reload'
  overlay.addEventListener('click', () => location.reload())
  container.style.position = 'relative'
  container.appendChild(overlay)
  return overlay
}

// ---------------------------------------------------------------------------
// Program wrapper
// ---------------------------------------------------------------------------

class Program {
  program: WebGLProgram
  uniforms: Record<string, WebGLUniformLocation>
  constructor(private gl: WebGLRenderingContext, vertSrc: string, fragSrc: string) {
    this.program = createProgram(gl, vertSrc, fragSrc)
    this.uniforms = getUniforms(gl, this.program)
  }
  bind() {
    this.gl.useProgram(this.program)
  }
}

// ---------------------------------------------------------------------------
// Main simulation class
// ---------------------------------------------------------------------------

export class FluidSimulation {
  private canvas: HTMLCanvasElement
  private gl!: WebGLRenderingContext
  private ext: {
    halfFloatType: number
    supportLinearFiltering: boolean
  } = { halfFloatType: 0, supportLinearFiltering: false }

  // Programs
  private splatProgram!: Program
  private advectionProgram!: Program
  private divergenceProgram!: Program
  private pressureProgram!: Program
  private gradientSubProgram!: Program
  private curlProgram!: Program
  private vorticityProgram!: Program
  private displayProgram!: Program
  private clearProgram!: Program

  // Framebuffers
  private velocity!: DoubleFBO
  private pressure!: DoubleFBO
  private divergenceFBO!: FBOHandle
  private curlFBO!: FBOHandle
  private dye!: DoubleFBO

  // Geometry
  private quadVAO!: WebGLBuffer

  // Dimensions
  private simWidth = 0
  private simHeight = 0
  private dyeWidth = 0
  private dyeHeight = 0

  // Config
  private config: FluidConfig = {
    viscosity: 0.3,
    curl: 30,
    pressure: 30,
    splatRadius: 0.25,
  }

  // Mouse
  private pointer = {
    x: 0, y: 0,
    prevX: 0, prevY: 0,
    dx: 0, dy: 0,
    down: false,
    moved: false,
    color: [0.5, 0.0, 0.8] as [number, number, number],
  }

  // Auto-splat
  private lastInteraction = 0
  private autoSplatTimer = 0
  private time = 0

  private animationId = 0
  private disposed = false
  private resizeObserver: ResizeObserver | null = null

  constructor(private container: HTMLDivElement) {
    this.canvas = document.createElement('canvas')
    this.canvas.style.width = '100%'
    this.canvas.style.height = '100%'
    this.canvas.style.display = 'block'
    container.appendChild(this.canvas)

    this.initGL()

    this.canvas.addEventListener('webglcontextlost', this.handleContextLost)
    this.canvas.addEventListener('webglcontextrestored', this.handleContextRestored)

    this.initPrograms()
    this.initQuad()
    this.initFramebuffers()
    this.initListeners()
    this.lastInteraction = performance.now()
    this.animate()
  }

  // --- Public API ---

  setConfig(partial: Partial<FluidConfig>) {
    Object.assign(this.config, partial)
  }

  getConfig(): FluidConfig {
    return { ...this.config }
  }

  splat(x: number, y: number, dx: number, dy: number, color?: [number, number, number]) {
    const c = color ?? this.randomColor()
    this.splatAtPoint(x, y, dx, dy, c)
  }

  resize() {
    this.resizeCanvas()
    this.initFramebuffers()
  }

  dispose() {
    this.disposed = true
    cancelAnimationFrame(this.animationId)
    this.removeListeners()
    this.canvas.removeEventListener('webglcontextlost', this.handleContextLost)
    this.canvas.removeEventListener('webglcontextrestored', this.handleContextRestored)
    if (this.resizeObserver) {
      this.resizeObserver.disconnect()
      this.resizeObserver = null
    }

    // Explicit WebGL resource cleanup
    const gl = this.gl

    const deleteFBO = (handle: FBOHandle) => {
      gl.deleteTexture(handle.texture)
      gl.deleteFramebuffer(handle.fbo)
    }
    const deleteDoubleFBO = (dfbo: DoubleFBO) => {
      deleteFBO(dfbo.read)
      deleteFBO(dfbo.write)
    }

    deleteDoubleFBO(this.velocity)
    deleteDoubleFBO(this.pressure)
    deleteDoubleFBO(this.dye)
    deleteFBO(this.divergenceFBO)
    deleteFBO(this.curlFBO)

    gl.deleteBuffer(this.quadVAO)

    const programs = [
      this.splatProgram, this.advectionProgram, this.divergenceProgram,
      this.pressureProgram, this.gradientSubProgram, this.curlProgram,
      this.vorticityProgram, this.displayProgram, this.clearProgram,
    ]
    for (const p of programs) {
      gl.deleteProgram(p.program)
    }

    gl.getExtension('WEBGL_lose_context')?.loseContext()
    if (this.canvas.parentElement) {
      this.canvas.parentElement.removeChild(this.canvas)
    }
  }

  // --- Init ---

  private initGL() {
    const params: WebGLContextAttributes = {
      alpha: true,
      depth: false,
      stencil: false,
      antialias: false,
      preserveDrawingBuffer: false,
    }
    const gl = this.canvas.getContext('webgl', params) || this.canvas.getContext('experimental-webgl', params) as WebGLRenderingContext
    if (!gl) throw new Error('WebGL not supported')
    this.gl = gl

    // Extensions
    const halfFloat = gl.getExtension('OES_texture_half_float')
    const halfFloatLinear = gl.getExtension('OES_texture_half_float_linear')
    const floatExt = gl.getExtension('OES_texture_float')
    const floatLinear = gl.getExtension('OES_texture_float_linear')

    let halfFloatType: number
    let supportLinearFiltering: boolean

    if (halfFloat) {
      halfFloatType = halfFloat.HALF_FLOAT_OES
      supportLinearFiltering = !!halfFloatLinear
    } else if (floatExt) {
      halfFloatType = gl.FLOAT
      supportLinearFiltering = !!floatLinear
    } else {
      halfFloatType = gl.UNSIGNED_BYTE
      supportLinearFiltering = true
    }

    this.ext = { halfFloatType, supportLinearFiltering }

    this.resizeCanvas()
  }

  private resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio, 2)
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    this.canvas.width = w * dpr
    this.canvas.height = h * dpr

    // Sim at half resolution for performance
    this.simWidth = Math.floor(w * dpr * 0.5)
    this.simHeight = Math.floor(h * dpr * 0.5)
    // Dye at 3/4 resolution for sharper colors
    this.dyeWidth = Math.floor(w * dpr * 0.75)
    this.dyeHeight = Math.floor(h * dpr * 0.75)
  }

  private initPrograms() {
    const gl = this.gl
    this.splatProgram = new Program(gl, BASE_VERT, SPLAT_FRAG)
    this.advectionProgram = new Program(gl, BASE_VERT, ADVECTION_FRAG)
    this.divergenceProgram = new Program(gl, BASE_VERT, DIVERGENCE_FRAG)
    this.pressureProgram = new Program(gl, BASE_VERT, PRESSURE_FRAG)
    this.gradientSubProgram = new Program(gl, BASE_VERT, GRADIENT_SUB_FRAG)
    this.curlProgram = new Program(gl, BASE_VERT, CURL_FRAG)
    this.vorticityProgram = new Program(gl, BASE_VERT, VORTICITY_FRAG)
    this.displayProgram = new Program(gl, BASE_VERT, DISPLAY_FRAG)
    this.clearProgram = new Program(gl, BASE_VERT, CLEAR_FRAG)
  }

  private initQuad() {
    const gl = this.gl
    const buf = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW)
    this.quadVAO = buf
  }

  private initFramebuffers() {
    const gl = this.gl
    const simW = this.simWidth
    const simH = this.simHeight
    const dyeW = this.dyeWidth
    const dyeH = this.dyeHeight

    const texType = this.ext.halfFloatType
    const filtering = this.ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST

    this.velocity = this.createDoubleFBO(simW, simH, gl.RGBA, texType, filtering)
    this.pressure = this.createDoubleFBO(simW, simH, gl.RGBA, texType, gl.NEAREST)
    this.divergenceFBO = this.createFBO(simW, simH, gl.RGBA, texType, gl.NEAREST)
    this.curlFBO = this.createFBO(simW, simH, gl.RGBA, texType, gl.NEAREST)
    this.dye = this.createDoubleFBO(dyeW, dyeH, gl.RGBA, texType, filtering)
  }

  private createFBO(w: number, h: number, internalFormat: number, type: number, filter: number): FBOHandle {
    const gl = this.gl
    const texture = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, gl.RGBA, type, null)

    const fbo = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0)
    gl.viewport(0, 0, w, h)
    gl.clear(gl.COLOR_BUFFER_BIT)

    return { texture, fbo, width: w, height: h }
  }

  private createDoubleFBO(w: number, h: number, internalFormat: number, type: number, filter: number): DoubleFBO {
    let fbo1 = this.createFBO(w, h, internalFormat, type, filter)
    let fbo2 = this.createFBO(w, h, internalFormat, type, filter)
    return {
      get read() { return fbo1 },
      get write() { return fbo2 },
      swap() { const t = fbo1; fbo1 = fbo2; fbo2 = t },
    }
  }

  // --- Drawing helpers ---

  private blit(dest: FBOHandle | null) {
    const gl = this.gl
    if (dest) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, dest.fbo)
      gl.viewport(0, 0, dest.width, dest.height)
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      gl.viewport(0, 0, this.canvas.width, this.canvas.height)
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVAO)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    gl.enableVertexAttribArray(0)
    gl.drawArrays(gl.TRIANGLE_FAN, 0, 4)
  }

  // --- Simulation steps ---

  private step(dt: number) {
    const gl = this.gl

    // Dissipation scales with viscosity (lower viscosity = more dissipation for interesting flow)
    const velocityDissipation = 1.0 - this.config.viscosity * 0.02
    const dyeDissipation = 0.97 - this.config.viscosity * 0.005

    // 1. Curl
    this.curlProgram.bind()
    gl.uniform2f(this.curlProgram.uniforms['texelSize'], 1.0 / this.simWidth, 1.0 / this.simHeight)
    gl.uniform1i(this.curlProgram.uniforms['uVelocity'], 0)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture)
    this.blit(this.curlFBO)

    // 2. Vorticity confinement
    this.vorticityProgram.bind()
    gl.uniform2f(this.vorticityProgram.uniforms['texelSize'], 1.0 / this.simWidth, 1.0 / this.simHeight)
    gl.uniform1i(this.vorticityProgram.uniforms['uVelocity'], 0)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture)
    gl.uniform1i(this.vorticityProgram.uniforms['uCurl'], 1)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, this.curlFBO.texture)
    gl.uniform1f(this.vorticityProgram.uniforms['curl'], this.config.curl)
    gl.uniform1f(this.vorticityProgram.uniforms['dt'], dt)
    this.blit(this.velocity.write)
    this.velocity.swap()

    // 3. Advect velocity
    this.advectionProgram.bind()
    gl.uniform2f(this.advectionProgram.uniforms['texelSize'], 1.0 / this.simWidth, 1.0 / this.simHeight)
    gl.uniform1i(this.advectionProgram.uniforms['uVelocity'], 0)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture)
    gl.uniform1i(this.advectionProgram.uniforms['uSource'], 0)
    gl.uniform1f(this.advectionProgram.uniforms['dt'], dt)
    gl.uniform1f(this.advectionProgram.uniforms['dissipation'], velocityDissipation)
    this.blit(this.velocity.write)
    this.velocity.swap()

    // 4. Advect dye
    this.advectionProgram.bind()
    gl.uniform2f(this.advectionProgram.uniforms['texelSize'], 1.0 / this.simWidth, 1.0 / this.simHeight)
    gl.uniform1i(this.advectionProgram.uniforms['uVelocity'], 0)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture)
    gl.uniform1i(this.advectionProgram.uniforms['uSource'], 1)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, this.dye.read.texture)
    gl.uniform1f(this.advectionProgram.uniforms['dissipation'], dyeDissipation)
    this.blit(this.dye.write)
    this.dye.swap()

    // 5. Divergence
    this.divergenceProgram.bind()
    gl.uniform2f(this.divergenceProgram.uniforms['texelSize'], 1.0 / this.simWidth, 1.0 / this.simHeight)
    gl.uniform1i(this.divergenceProgram.uniforms['uVelocity'], 0)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture)
    this.blit(this.divergenceFBO)

    // 6. Clear pressure
    this.clearProgram.bind()
    gl.uniform1i(this.clearProgram.uniforms['uTexture'], 0)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.texture)
    gl.uniform1f(this.clearProgram.uniforms['value'], 0.8)
    this.blit(this.pressure.write)
    this.pressure.swap()

    // 7. Pressure solve (Jacobi iterations)
    this.pressureProgram.bind()
    gl.uniform2f(this.pressureProgram.uniforms['texelSize'], 1.0 / this.simWidth, 1.0 / this.simHeight)
    gl.uniform1i(this.pressureProgram.uniforms['uDivergence'], 1)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, this.divergenceFBO.texture)
    for (let i = 0; i < this.config.pressure; i++) {
      gl.uniform1i(this.pressureProgram.uniforms['uPressure'], 0)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.texture)
      this.blit(this.pressure.write)
      this.pressure.swap()
    }

    // 8. Subtract pressure gradient
    this.gradientSubProgram.bind()
    gl.uniform2f(this.gradientSubProgram.uniforms['texelSize'], 1.0 / this.simWidth, 1.0 / this.simHeight)
    gl.uniform1i(this.gradientSubProgram.uniforms['uPressure'], 0)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.texture)
    gl.uniform1i(this.gradientSubProgram.uniforms['uVelocity'], 1)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture)
    this.blit(this.velocity.write)
    this.velocity.swap()
  }

  private splatAtPoint(x: number, y: number, dx: number, dy: number, color: [number, number, number]) {
    const gl = this.gl

    // Splat velocity
    this.splatProgram.bind()
    gl.uniform1i(this.splatProgram.uniforms['uTarget'], 0)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture)
    gl.uniform1f(this.splatProgram.uniforms['aspectRatio'], this.canvas.width / this.canvas.height)
    gl.uniform2f(this.splatProgram.uniforms['point'], x, y)
    gl.uniform3f(this.splatProgram.uniforms['color'], dx, dy, 0.0)
    gl.uniform1f(this.splatProgram.uniforms['radius'], this.correctRadius(this.config.splatRadius * 0.01))
    this.blit(this.velocity.write)
    this.velocity.swap()

    // Splat dye
    gl.uniform1i(this.splatProgram.uniforms['uTarget'], 0)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.dye.read.texture)
    gl.uniform3f(this.splatProgram.uniforms['color'], color[0], color[1], color[2])
    gl.uniform1f(this.splatProgram.uniforms['radius'], this.correctRadius(this.config.splatRadius * 0.01))
    this.blit(this.dye.write)
    this.dye.swap()
  }

  private correctRadius(radius: number): number {
    const aspectRatio = this.canvas.width / this.canvas.height
    if (aspectRatio > 1) {
      return radius * aspectRatio
    }
    return radius
  }

  private display() {
    const gl = this.gl
    this.displayProgram.bind()
    gl.uniform1i(this.displayProgram.uniforms['uTexture'], 0)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.dye.read.texture)
    this.blit(null) // draw to screen
  }

  // --- Color ---

  private hue = 0
  private randomColor(): [number, number, number] {
    this.hue += 0.03 + Math.random() * 0.05
    if (this.hue > 1) this.hue -= 1
    return this.hsvToRgb(this.hue, 1.0, 1.0)
  }

  private mouseColor(): [number, number, number] {
    // Color based on mouse position + cycling
    const t = performance.now() * 0.001
    const h = (this.pointer.x * 0.3 + this.pointer.y * 0.3 + t * 0.1) % 1
    return this.hsvToRgb(h < 0 ? h + 1 : h, 1.0, 1.0)
  }

  private hsvToRgb(h: number, s: number, v: number): [number, number, number] {
    const i = Math.floor(h * 6)
    const f = h * 6 - i
    const p = v * (1 - s)
    const q = v * (1 - f * s)
    const t = v * (1 - (1 - f) * s)
    let r: number, g: number, b: number
    switch (i % 6) {
      case 0: r = v; g = t; b = p; break
      case 1: r = q; g = v; b = p; break
      case 2: r = p; g = v; b = t; break
      case 3: r = p; g = q; b = v; break
      case 4: r = t; g = p; b = v; break
      case 5: r = v; g = p; b = q; break
      default: r = 0; g = 0; b = 0
    }
    return [r, g, b]
  }

  // --- Auto splats ---

  private generateAutoSplat() {
    const x = Math.random()
    const y = Math.random()
    const angle = Math.random() * Math.PI * 2
    const speed = 80 + Math.random() * 200
    const dx = Math.cos(angle) * speed
    const dy = Math.sin(angle) * speed
    const color = this.randomColor()
    this.splatAtPoint(x, y, dx, dy, color)
  }

  private generateMultipleSplats(count: number) {
    for (let i = 0; i < count; i++) {
      this.generateAutoSplat()
    }
  }

  // --- Input ---

  private onMouseDown = (e: MouseEvent) => {
    this.pointer.down = true
    this.updatePointerPosition(e.clientX, e.clientY)
    this.pointer.prevX = this.pointer.x
    this.pointer.prevY = this.pointer.y
    this.lastInteraction = performance.now()
  }

  private onMouseMove = (e: MouseEvent) => {
    this.pointer.prevX = this.pointer.x
    this.pointer.prevY = this.pointer.y
    this.updatePointerPosition(e.clientX, e.clientY)
    this.pointer.moved = true
    this.lastInteraction = performance.now()
  }

  private onMouseUp = () => {
    this.pointer.down = false
  }

  private onTouchStart = (e: TouchEvent) => {
    e.preventDefault()
    const touch = e.touches[0]
    this.pointer.down = true
    this.updatePointerPosition(touch.clientX, touch.clientY)
    this.pointer.prevX = this.pointer.x
    this.pointer.prevY = this.pointer.y
    this.lastInteraction = performance.now()
  }

  private onTouchMove = (e: TouchEvent) => {
    e.preventDefault()
    const touch = e.touches[0]
    this.pointer.prevX = this.pointer.x
    this.pointer.prevY = this.pointer.y
    this.updatePointerPosition(touch.clientX, touch.clientY)
    this.pointer.moved = true
    this.lastInteraction = performance.now()
  }

  private onTouchEnd = () => {
    this.pointer.down = false
  }

  private updatePointerPosition(clientX: number, clientY: number) {
    const rect = this.canvas.getBoundingClientRect()
    this.pointer.x = (clientX - rect.left) / rect.width
    this.pointer.y = 1.0 - (clientY - rect.top) / rect.height
  }

  private initListeners() {
    this.canvas.addEventListener('mousedown', this.onMouseDown)
    this.canvas.addEventListener('mousemove', this.onMouseMove)
    window.addEventListener('mouseup', this.onMouseUp)
    this.canvas.addEventListener('touchstart', this.onTouchStart, { passive: false })
    this.canvas.addEventListener('touchmove', this.onTouchMove, { passive: false })
    window.addEventListener('touchend', this.onTouchEnd)

    this.resizeObserver = new ResizeObserver(() => {
      if (!this.disposed) this.resize()
    })
    this.resizeObserver.observe(this.container)

    // Initial splats for visual interest
    setTimeout(() => {
      if (!this.disposed) this.generateMultipleSplats(5)
    }, 100)
  }

  private removeListeners() {
    this.canvas.removeEventListener('mousedown', this.onMouseDown)
    this.canvas.removeEventListener('mousemove', this.onMouseMove)
    window.removeEventListener('mouseup', this.onMouseUp)
    this.canvas.removeEventListener('touchstart', this.onTouchStart)
    this.canvas.removeEventListener('touchmove', this.onTouchMove)
    window.removeEventListener('touchend', this.onTouchEnd)
  }

  // --- Context loss ---

  private handleContextLost = (e: Event) => {
    e.preventDefault()
    cancelAnimationFrame(this.animationId)
    showContextLostOverlay(this.container)
  }

  private handleContextRestored = () => {
    location.reload()
  }

  // --- Main loop ---

  private lastTime = 0

  private animate = () => {
    if (this.disposed) return
    this.animationId = requestAnimationFrame(this.animate)

    const now = performance.now()
    const dt = Math.min((now - (this.lastTime || now)) * 0.001, 0.016667)
    this.lastTime = now
    this.time += dt

    // Process mouse input
    if (this.pointer.moved) {
      this.pointer.moved = false
      const dx = (this.pointer.x - this.pointer.prevX) * this.canvas.width
      const dy = (this.pointer.y - this.pointer.prevY) * this.canvas.height
      const speed = Math.sqrt(dx * dx + dy * dy)

      if (this.pointer.down && speed > 0.5) {
        const color = this.mouseColor()
        const scaledDx = dx * 10
        const scaledDy = dy * 10
        this.splatAtPoint(this.pointer.x, this.pointer.y, scaledDx, scaledDy, color)
      }
    }

    // Auto splats when idle
    const idleTime = now - this.lastInteraction
    if (idleTime > 3000) {
      this.autoSplatTimer += dt
      if (this.autoSplatTimer > 0.4) {
        this.autoSplatTimer = 0
        this.generateAutoSplat()
      }
    } else {
      this.autoSplatTimer = 0
    }

    this.step(dt)
    this.display()
  }
}
