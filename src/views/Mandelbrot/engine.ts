/**
 * Mandelbrot Explorer Engine
 *
 * Pure WebGL renderer for the Mandelbrot set with:
 * - Smooth iteration coloring
 * - Multiple color palettes
 * - Double-emulated precision for deep zooms
 * - Mouse wheel zoom, drag to pan, pinch-to-zoom on touch
 */

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

function createProgram(gl: WebGLRenderingContext, vertSrc: string, fragSrc: string): WebGLProgram {
  const program = gl.createProgram()
  if (!program) throw new Error('Failed to create program — WebGL context may be lost')
  gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, vertSrc))
  gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, fragSrc))
  gl.bindAttribLocation(program, 0, 'aPosition')
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`Program link error: ${gl.getProgramInfoLog(program)}`)
  }
  return program
}

/** Show a context-lost overlay on the container */
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
// Shaders
// ---------------------------------------------------------------------------

const VERT_SRC = `
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`

// Double-emulated precision Mandelbrot fragment shader.
// Uses vec2 (hi, lo) pairs to emulate double-precision arithmetic,
// allowing deeper zoom levels than float32 alone.
const FRAG_SRC = `
precision highp float;
varying vec2 vUv;

uniform vec2 uCenterHi;   // high part of center (complex plane)
uniform vec2 uCenterLo;   // low part of center (for double emulation)
uniform float uZoom;       // zoom level (half-width of view in complex plane)
uniform int uMaxIter;
uniform int uPalette;
uniform vec2 uResolution;

// Double-float add: (a_hi, a_lo) + (b_hi, b_lo) = (s_hi, s_lo)
vec2 df_add(vec2 a, vec2 b) {
  float s = a.x + b.x;
  float e = s - a.x;
  float t = ((b.x - e) + (a.x - (s - e))) + a.y + b.y;
  return vec2(s + t, t - ((s + t) - s));
}

// Double-float multiply: (a_hi, a_lo) * (b_hi, b_lo) = (p_hi, p_lo)
vec2 df_mul(vec2 a, vec2 b) {
  float p = a.x * b.x;
  float e = a.x * b.x - p; // fma approximation
  // For WebGL without fma, use split approach
  float t = a.x * b.y + a.y * b.x + e;
  return vec2(p + t, t - ((p + t) - p));
}

// Palette functions
vec3 palette0(float t) {
  // Ultra Fractal classic: blue-white-orange-black
  vec3 a = vec3(0.5, 0.5, 0.5);
  vec3 b = vec3(0.5, 0.5, 0.5);
  vec3 c = vec3(1.0, 1.0, 1.0);
  vec3 d = vec3(0.0, 0.10, 0.20);
  return a + b * cos(6.28318 * (c * t + d));
}

vec3 palette1(float t) {
  // Fire: black-red-orange-yellow-white
  vec3 a = vec3(0.5, 0.5, 0.5);
  vec3 b = vec3(0.5, 0.5, 0.5);
  vec3 c = vec3(1.0, 1.0, 0.5);
  vec3 d = vec3(0.80, 0.90, 0.30);
  return a + b * cos(6.28318 * (c * t + d));
}

vec3 palette2(float t) {
  // Ocean: deep blue to cyan to white
  vec3 a = vec3(0.5, 0.5, 0.5);
  vec3 b = vec3(0.5, 0.5, 0.5);
  vec3 c = vec3(1.0, 0.7, 0.4);
  vec3 d = vec3(0.0, 0.15, 0.20);
  return a + b * cos(6.28318 * (c * t + d));
}

vec3 palette3(float t) {
  // Neon: magenta-cyan-green
  vec3 a = vec3(0.5, 0.5, 0.5);
  vec3 b = vec3(0.5, 0.5, 0.5);
  vec3 c = vec3(2.0, 1.0, 0.0);
  vec3 d = vec3(0.50, 0.20, 0.25);
  return a + b * cos(6.28318 * (c * t + d));
}

vec3 palette4(float t) {
  // Grayscale with blue tint
  float v = t;
  return vec3(v * 0.85, v * 0.9, v);
}

vec3 getColor(float t, int pal) {
  if (pal == 0) return palette0(t);
  if (pal == 1) return palette1(t);
  if (pal == 2) return palette2(t);
  if (pal == 3) return palette3(t);
  return palette4(t);
}

void main() {
  float aspect = uResolution.x / uResolution.y;
  vec2 screenPos = (vUv - 0.5) * 2.0;
  screenPos.x *= aspect;

  // Map screen position to complex plane using double-emulated precision
  // c = center + screenPos * zoom
  vec2 cxHi = df_add(vec2(uCenterHi.x, uCenterLo.x), vec2(screenPos.x * uZoom, 0.0));
  vec2 cyHi = df_add(vec2(uCenterHi.y, uCenterLo.y), vec2(screenPos.y * uZoom, 0.0));

  // Mandelbrot iteration using double-emulated precision
  // z = 0, iterate z = z^2 + c
  vec2 zxHi = vec2(0.0, 0.0);
  vec2 zyHi = vec2(0.0, 0.0);

  float escaped = 0.0;
  float smoothIter = 0.0;

  // Use standard float for performance at moderate zoom levels,
  // switch to double-emulated when needed
  bool useDouble = uZoom < 0.00001;

  if (useDouble) {
    // Double-emulated precision path
    for (int i = 0; i < 1000; i++) {
      if (i >= uMaxIter) break;

      // zx^2, zy^2
      vec2 zx2 = df_mul(zxHi, zxHi);
      vec2 zy2 = df_mul(zyHi, zyHi);

      // Bailout check: |z|^2 > 4.0
      float mag2 = zx2.x + zy2.x;
      if (mag2 > 256.0) {
        escaped = 1.0;
        // Smooth coloring: n - log2(log2(|z|))
        float logZn = log(mag2) * 0.5;
        float nu = log(logZn) / log(2.0);
        smoothIter = float(i) + 1.0 - nu;
        break;
      }

      // z = z^2 + c
      // new_zx = zx^2 - zy^2 + cx
      // new_zy = 2*zx*zy + cy
      vec2 zxzy = df_mul(zxHi, zyHi);
      vec2 newZx = df_add(df_add(zx2, vec2(-zy2.x, -zy2.y)), cxHi);
      vec2 newZy = df_add(df_add(zxzy, zxzy), cyHi);

      zxHi = newZx;
      zyHi = newZy;
    }
  } else {
    // Standard float precision path (faster)
    float zx = 0.0;
    float zy = 0.0;
    float cx = cxHi.x;
    float cy = cyHi.x;

    for (int i = 0; i < 1000; i++) {
      if (i >= uMaxIter) break;

      float zx2 = zx * zx;
      float zy2 = zy * zy;

      if (zx2 + zy2 > 256.0) {
        escaped = 1.0;
        float logZn = log(zx2 + zy2) * 0.5;
        float nu = log(logZn) / log(2.0);
        smoothIter = float(i) + 1.0 - nu;
        break;
      }

      float newZx = zx2 - zy2 + cx;
      zy = 2.0 * zx * zy + cy;
      zx = newZx;
    }
  }

  vec3 col;
  if (escaped > 0.5) {
    float t = fract(smoothIter * 0.02);
    col = getColor(t, uPalette);
  } else {
    col = vec3(0.0);
  }

  gl_FragColor = vec4(col, 1.0);
}
`

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

export class MandelbrotEngine {
  private canvas: HTMLCanvasElement
  private gl: WebGLRenderingContext
  private program: WebGLProgram
  private quadBuffer: WebGLBuffer

  // Uniforms
  private uCenterHi: WebGLUniformLocation | null
  private uCenterLo: WebGLUniformLocation | null
  private uZoom: WebGLUniformLocation | null
  private uMaxIter: WebGLUniformLocation | null
  private uPalette: WebGLUniformLocation | null
  private uResolution: WebGLUniformLocation | null

  // State
  private _centerX = -0.5
  private _centerY = 0.0
  private _zoom = 1.5 // half-width in complex plane
  private _maxIter = 200
  private _palette = 0

  // Interaction
  private dragging = false
  private dragStartX = 0
  private dragStartY = 0
  private dragCenterStartX = 0
  private dragCenterStartY = 0

  // Touch
  private lastPinchDist = 0
  private touchStartX = 0
  private touchStartY = 0
  private touchCenterStartX = 0
  private touchCenterStartY = 0

  private animationId = 0
  private needsRender = true
  private disposed = false
  private resizeObserver: ResizeObserver | null = null

  // Callback for view changes (H-7 fix: replaces rAF polling in React)
  onViewChange: ((cx: number, cy: number, zoom: number) => void) | null = null

  constructor(private container: HTMLDivElement) {
    this.canvas = document.createElement('canvas')
    this.canvas.style.width = '100%'
    this.canvas.style.height = '100%'
    this.canvas.style.display = 'block'
    this.canvas.style.cursor = 'grab'
    container.appendChild(this.canvas)

    const gl = this.canvas.getContext('webgl', {
      alpha: false,
      depth: false,
      stencil: false,
      antialias: false,
      preserveDrawingBuffer: false,
    })
    if (!gl) throw new Error('WebGL not supported')
    this.gl = gl

    this.program = createProgram(gl, VERT_SRC, FRAG_SRC)

    // Get uniform locations
    this.uCenterHi = gl.getUniformLocation(this.program, 'uCenterHi')
    this.uCenterLo = gl.getUniformLocation(this.program, 'uCenterLo')
    this.uZoom = gl.getUniformLocation(this.program, 'uZoom')
    this.uMaxIter = gl.getUniformLocation(this.program, 'uMaxIter')
    this.uPalette = gl.getUniformLocation(this.program, 'uPalette')
    this.uResolution = gl.getUniformLocation(this.program, 'uResolution')

    // Quad buffer
    this.quadBuffer = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW)

    // C-1 fix: WebGL context loss handling
    this.canvas.addEventListener('webglcontextlost', this.handleContextLost)
    this.canvas.addEventListener('webglcontextrestored', this.handleContextRestored)

    this.resizeCanvas()
    this.initListeners()
    this.animate()
  }

  private handleContextLost = (e: Event) => {
    e.preventDefault()
    cancelAnimationFrame(this.animationId)
    showContextLostOverlay(this.container)
  }

  private handleContextRestored = () => {
    // Context restored — reload page for clean state
    location.reload()
  }

  // --- Public API ---

  get center(): [number, number] {
    return [this._centerX, this._centerY]
  }

  get zoom(): number {
    return this._zoom
  }

  zoomAt(screenX: number, screenY: number, factor: number) {
    // Convert screen position to complex plane coordinates
    const rect = this.canvas.getBoundingClientRect()
    const aspect = this.canvas.width / this.canvas.height
    const normX = ((screenX - rect.left) / rect.width - 0.5) * 2.0 * aspect
    const normY = -((screenY - rect.top) / rect.height - 0.5) * 2.0

    // Position in complex plane
    const cx = this._centerX + normX * this._zoom
    const cy = this._centerY + normY * this._zoom

    // Zoom
    this._zoom *= factor

    // Adjust center to keep the zoom point fixed
    this._centerX = cx - normX * this._zoom
    this._centerY = cy - normY * this._zoom

    this.needsRender = true
  }

  pan(dx: number, dy: number) {
    const aspect = this.canvas.width / this.canvas.height
    const rect = this.canvas.getBoundingClientRect()
    this._centerX -= (dx / rect.width) * 2.0 * this._zoom * aspect
    this._centerY += (dy / rect.height) * 2.0 * this._zoom
    this.needsRender = true
  }

  resetView() {
    this._centerX = -0.5
    this._centerY = 0.0
    this._zoom = 1.5
    this._maxIter = 200
    this.needsRender = true
  }

  setPalette(index: number) {
    this._palette = Math.max(0, Math.min(4, index))
    this.needsRender = true
  }

  setMaxIterations(n: number) {
    this._maxIter = Math.max(50, Math.min(1000, n))
    this.needsRender = true
  }

  get maxIterations(): number {
    return this._maxIter
  }

  get palette(): number {
    return this._palette
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
    gl.deleteProgram(this.program)
    gl.deleteBuffer(this.quadBuffer)
    gl.getExtension('WEBGL_lose_context')?.loseContext()
    if (this.canvas.parentElement) {
      this.canvas.parentElement.removeChild(this.canvas)
    }
  }

  // --- Internal ---

  private resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio, 2)
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    this.canvas.width = Math.floor(w * dpr)
    this.canvas.height = Math.floor(h * dpr)
    this.needsRender = true
  }

  // Split a float64 into hi + lo pair for double-emulated precision
  private splitDouble(value: number): [number, number] {
    const hi = Math.fround(value)
    const lo = value - hi
    return [hi, lo]
  }

  private render() {
    const gl = this.gl
    gl.viewport(0, 0, this.canvas.width, this.canvas.height)
    gl.useProgram(this.program)

    // Split center into hi/lo for double-emulated precision
    const [cxHi, cxLo] = this.splitDouble(this._centerX)
    const [cyHi, cyLo] = this.splitDouble(this._centerY)

    gl.uniform2f(this.uCenterHi, cxHi, cyHi)
    gl.uniform2f(this.uCenterLo, cxLo, cyLo)
    gl.uniform1f(this.uZoom, this._zoom)
    gl.uniform1i(this.uMaxIter, this._maxIter)
    gl.uniform1i(this.uPalette, this._palette)
    gl.uniform2f(this.uResolution, this.canvas.width, this.canvas.height)

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    gl.drawArrays(gl.TRIANGLE_FAN, 0, 4)
  }

  // --- Listeners ---

  private onWheel = (e: WheelEvent) => {
    e.preventDefault()
    const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15
    this.zoomAt(e.clientX, e.clientY, factor)
  }

  private onMouseDown = (e: MouseEvent) => {
    if (e.button !== 0) return
    this.dragging = true
    this.dragStartX = e.clientX
    this.dragStartY = e.clientY
    this.dragCenterStartX = this._centerX
    this.dragCenterStartY = this._centerY
    this.canvas.style.cursor = 'grabbing'
  }

  private onMouseMove = (e: MouseEvent) => {
    if (!this.dragging) return
    const dx = e.clientX - this.dragStartX
    const dy = e.clientY - this.dragStartY
    const rect = this.canvas.getBoundingClientRect()
    const aspect = this.canvas.width / this.canvas.height
    this._centerX = this.dragCenterStartX - (dx / rect.width) * 2.0 * this._zoom * aspect
    this._centerY = this.dragCenterStartY + (dy / rect.height) * 2.0 * this._zoom
    this.needsRender = true
  }

  private onMouseUp = () => {
    this.dragging = false
    this.canvas.style.cursor = 'grab'
  }

  private onTouchStart = (e: TouchEvent) => {
    e.preventDefault()
    if (e.touches.length === 1) {
      const t = e.touches[0]
      this.touchStartX = t.clientX
      this.touchStartY = t.clientY
      this.touchCenterStartX = this._centerX
      this.touchCenterStartY = this._centerY
      this.dragging = true
    } else if (e.touches.length === 2) {
      this.dragging = false
      const t0 = e.touches[0]
      const t1 = e.touches[1]
      this.lastPinchDist = Math.hypot(t1.clientX - t0.clientX, t1.clientY - t0.clientY)
    }
  }

  private onTouchMove = (e: TouchEvent) => {
    e.preventDefault()
    if (e.touches.length === 1 && this.dragging) {
      const t = e.touches[0]
      const dx = t.clientX - this.touchStartX
      const dy = t.clientY - this.touchStartY
      const rect = this.canvas.getBoundingClientRect()
      const aspect = this.canvas.width / this.canvas.height
      this._centerX = this.touchCenterStartX - (dx / rect.width) * 2.0 * this._zoom * aspect
      this._centerY = this.touchCenterStartY + (dy / rect.height) * 2.0 * this._zoom
      this.needsRender = true
    } else if (e.touches.length === 2) {
      const t0 = e.touches[0]
      const t1 = e.touches[1]
      const dist = Math.hypot(t1.clientX - t0.clientX, t1.clientY - t0.clientY)
      if (this.lastPinchDist > 0) {
        const midX = (t0.clientX + t1.clientX) * 0.5
        const midY = (t0.clientY + t1.clientY) * 0.5
        const factor = this.lastPinchDist / dist
        this.zoomAt(midX, midY, factor)
      }
      this.lastPinchDist = dist
    }
  }

  private onTouchEnd = (e: TouchEvent) => {
    if (e.touches.length < 2) {
      this.lastPinchDist = 0
    }
    if (e.touches.length === 0) {
      this.dragging = false
    } else if (e.touches.length === 1) {
      // Switch from pinch back to drag
      const t = e.touches[0]
      this.touchStartX = t.clientX
      this.touchStartY = t.clientY
      this.touchCenterStartX = this._centerX
      this.touchCenterStartY = this._centerY
      this.dragging = true
    }
  }

  private initListeners() {
    this.canvas.addEventListener('wheel', this.onWheel, { passive: false })
    this.canvas.addEventListener('mousedown', this.onMouseDown)
    window.addEventListener('mousemove', this.onMouseMove)
    window.addEventListener('mouseup', this.onMouseUp)
    this.canvas.addEventListener('touchstart', this.onTouchStart, { passive: false })
    this.canvas.addEventListener('touchmove', this.onTouchMove, { passive: false })
    this.canvas.addEventListener('touchend', this.onTouchEnd)

    this.resizeObserver = new ResizeObserver(() => {
      if (!this.disposed) {
        this.resizeCanvas()
      }
    })
    this.resizeObserver.observe(this.container)
  }

  private removeListeners() {
    this.canvas.removeEventListener('wheel', this.onWheel)
    this.canvas.removeEventListener('mousedown', this.onMouseDown)
    window.removeEventListener('mousemove', this.onMouseMove)
    window.removeEventListener('mouseup', this.onMouseUp)
    this.canvas.removeEventListener('touchstart', this.onTouchStart)
    this.canvas.removeEventListener('touchmove', this.onTouchMove)
    this.canvas.removeEventListener('touchend', this.onTouchEnd)
  }

  // --- Animation loop (only renders when needed) ---

  private animate = () => {
    if (this.disposed) return
    this.animationId = requestAnimationFrame(this.animate)

    if (this.needsRender) {
      this.needsRender = false
      this.render()
      this.onViewChange?.(this._centerX, this._centerY, this._zoom)
    }
  }
}
