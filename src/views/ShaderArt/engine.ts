/**
 * Shader Art Gallery Engine
 *
 * Pure WebGL renderer that displays a full-screen quad with
 * interchangeable fragment shaders. Each shader receives
 * uTime, uResolution, and uMouse uniforms.
 */

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function compileShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!
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
  const program = gl.createProgram()!
  gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, vertSrc))
  gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, fragSrc))
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`Program link error: ${gl.getProgramInfoLog(program)}`)
  }
  return program
}

// ---------------------------------------------------------------------------
// Vertex shader (shared)
// ---------------------------------------------------------------------------

const VERT_SRC = `
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`

// ---------------------------------------------------------------------------
// Fragment shaders
// ---------------------------------------------------------------------------

const SHADER_NAMES = [
  'Plasma',
  'Kaleidoscope',
  'Metaballs',
  'Fractal Flames',
  'Voronoi Cells',
  'Aurora',
  'Liquid Metal',
  'Neon Grid',
]

const FRAG_PREFIX = `
precision highp float;
varying vec2 vUv;
uniform float uTime;
uniform vec2 uResolution;
uniform vec2 uMouse;
`

const FRAG_SHADERS: string[] = [
  // 0 - Plasma
  FRAG_PREFIX + `
void main() {
  vec2 uv = vUv;
  float t = uTime * 0.8;
  float v = 0.0;
  vec2 c = uv * 6.0 - 3.0;
  v += sin(c.x + t);
  v += sin((c.y + t) * 0.5);
  v += sin((c.x + c.y + t) * 0.5);
  c += vec2(sin(t * 0.3), cos(t * 0.5)) * 2.0;
  v += sin(sqrt(c.x * c.x + c.y * c.y + 1.0) + t);
  v *= 0.5;

  vec3 col = vec3(
    sin(v * 3.14159 + 0.0) * 0.5 + 0.5,
    sin(v * 3.14159 + 2.094) * 0.5 + 0.5,
    sin(v * 3.14159 + 4.188) * 0.5 + 0.5
  );

  // Mouse interaction: shift colors near cursor
  float md = length(uv - uMouse) * 3.0;
  col = mix(col.gbr, col, smoothstep(0.0, 1.0, md));

  gl_FragColor = vec4(col, 1.0);
}
`,

  // 1 - Kaleidoscope
  FRAG_PREFIX + `
void main() {
  vec2 uv = (vUv - 0.5) * 2.0;
  float aspect = uResolution.x / uResolution.y;
  uv.x *= aspect;

  float t = uTime * 0.5;
  float angle = atan(uv.y, uv.x) + t * 0.3;
  float r = length(uv);

  // Kaleidoscope fold
  float segments = 8.0;
  angle = mod(angle, 3.14159 * 2.0 / segments);
  angle = abs(angle - 3.14159 / segments);

  vec2 p = vec2(cos(angle), sin(angle)) * r;

  // Pattern
  float v = 0.0;
  v += sin(p.x * 8.0 + t * 2.0) * 0.5;
  v += cos(p.y * 6.0 - t * 1.5) * 0.5;
  v += sin((p.x + p.y) * 10.0 + t) * 0.3;
  v += sin(r * 12.0 - t * 3.0) * 0.4;

  vec3 col;
  col.r = sin(v * 3.0 + t) * 0.5 + 0.5;
  col.g = sin(v * 3.0 + t + 2.094) * 0.5 + 0.5;
  col.b = sin(v * 3.0 + t + 4.188) * 0.5 + 0.5;

  // Brighten center
  col *= 1.0 - r * 0.3;
  // Mouse shifts pattern
  col *= 1.0 + 0.3 * sin(r * 6.0 + uMouse.x * 6.28);

  gl_FragColor = vec4(col, 1.0);
}
`,

  // 2 - Metaballs
  FRAG_PREFIX + `
void main() {
  vec2 uv = vUv;
  float aspect = uResolution.x / uResolution.y;
  uv.x *= aspect;

  float t = uTime * 0.7;
  float sum = 0.0;

  // 6 metaballs
  for (int i = 0; i < 6; i++) {
    float fi = float(i);
    vec2 center = vec2(
      0.5 * aspect + sin(t * (0.3 + fi * 0.1) + fi * 1.2) * 0.35 * aspect,
      0.5 + cos(t * (0.4 + fi * 0.12) + fi * 0.8) * 0.35
    );
    float d = length(uv - center);
    sum += 0.015 / (d * d + 0.001);
  }

  // Mouse-controlled metaball
  vec2 mousePos = vec2(uMouse.x * aspect, uMouse.y);
  float md = length(uv - mousePos);
  sum += 0.02 / (md * md + 0.001);

  // Color based on field strength
  float edge = smoothstep(8.0, 12.0, sum);
  float glow = smoothstep(3.0, 8.0, sum);

  vec3 neon1 = vec3(0.0, 1.0, 0.8);
  vec3 neon2 = vec3(1.0, 0.0, 0.6);
  vec3 neon3 = vec3(0.2, 0.4, 1.0);

  vec3 col = mix(neon3, neon1, sin(sum * 0.3 + t) * 0.5 + 0.5);
  col = mix(col, neon2, sin(sum * 0.2 + t * 0.5 + 1.0) * 0.5 + 0.5);

  col *= glow;
  col += vec3(1.0) * edge * 0.3;

  // Subtle dark background
  col = mix(vec3(0.02, 0.01, 0.05), col, smoothstep(2.0, 5.0, sum));

  gl_FragColor = vec4(col, 1.0);
}
`,

  // 3 - Fractal Flames
  FRAG_PREFIX + `
vec2 variation1(vec2 p, float t) {
  return vec2(sin(p.x + t), sin(p.y + t));
}
vec2 variation2(vec2 p) {
  float r2 = dot(p, p);
  return p / (r2 + 0.001);
}
vec2 variation3(vec2 p, float t) {
  float r = length(p);
  float theta = atan(p.y, p.x);
  return r * vec2(sin(theta + r + t), cos(theta - r + t));
}

void main() {
  vec2 uv = (vUv - 0.5) * 4.0;
  float t = uTime * 0.3;

  vec3 accum = vec3(0.0);
  vec2 p = uv + (uMouse - 0.5) * 2.0;

  for (int i = 0; i < 20; i++) {
    float fi = float(i);
    float w = fract(fi * 0.618 + t * 0.1);

    if (w < 0.33) {
      p = variation1(p * 0.7, t + fi * 0.1);
    } else if (w < 0.66) {
      p = variation2(p * 0.8);
    } else {
      p = variation3(p * 0.6, t + fi * 0.05);
    }

    // Accumulate color
    float intensity = exp(-length(p) * 0.5) * 0.15;
    vec3 c = vec3(
      sin(fi * 0.3 + t) * 0.5 + 0.5,
      sin(fi * 0.3 + t + 2.0) * 0.5 + 0.5,
      sin(fi * 0.3 + t + 4.0) * 0.5 + 0.5
    );
    accum += c * intensity;
  }

  // Tone mapping
  accum = 1.0 - exp(-accum * 3.0);

  // Vignette
  float vig = 1.0 - dot(vUv - 0.5, vUv - 0.5) * 1.5;
  accum *= vig;

  gl_FragColor = vec4(accum, 1.0);
}
`,

  // 4 - Voronoi Cells
  FRAG_PREFIX + `
vec2 hash2(vec2 p) {
  p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
  return fract(sin(p) * 43758.5453);
}

void main() {
  vec2 uv = vUv;
  float aspect = uResolution.x / uResolution.y;
  uv.x *= aspect;
  float t = uTime * 0.5;

  float scale = 6.0;
  vec2 st = uv * scale;
  vec2 i_st = floor(st);
  vec2 f_st = fract(st);

  float minDist = 10.0;
  float secondDist = 10.0;
  vec2 minPoint = vec2(0.0);

  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 neighbor = vec2(float(x), float(y));
      vec2 point = hash2(i_st + neighbor);
      // Animate points
      point = 0.5 + 0.5 * sin(t + 6.2831 * point);
      vec2 diff = neighbor + point - f_st;
      float d = length(diff);
      if (d < minDist) {
        secondDist = minDist;
        minDist = d;
        minPoint = point;
      } else if (d < secondDist) {
        secondDist = d;
      }
    }
  }

  float edge = secondDist - minDist;

  // Coloring
  vec3 cellColor = vec3(
    sin(minPoint.x * 6.28 + t) * 0.5 + 0.5,
    sin(minPoint.y * 6.28 + t + 2.0) * 0.5 + 0.5,
    sin((minPoint.x + minPoint.y) * 6.28 + t + 4.0) * 0.5 + 0.5
  );

  // Edge glow
  float edgeLine = 1.0 - smoothstep(0.0, 0.08, edge);
  vec3 edgeColor = vec3(0.9, 0.95, 1.0);

  // Distance gradient inside cell
  cellColor *= 0.3 + 0.7 * (1.0 - minDist);

  vec3 col = mix(cellColor, edgeColor, edgeLine);

  // Mouse proximity brightening
  float mouseDist = length(vUv - uMouse);
  col += 0.15 * exp(-mouseDist * 4.0);

  gl_FragColor = vec4(col, 1.0);
}
`,

  // 5 - Aurora
  FRAG_PREFIX + `
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = sin(dot(i, vec2(127.1, 311.7))) * 43758.5453;
  float b = sin(dot(i + vec2(1.0, 0.0), vec2(127.1, 311.7))) * 43758.5453;
  float c = sin(dot(i + vec2(0.0, 1.0), vec2(127.1, 311.7))) * 43758.5453;
  float d = sin(dot(i + vec2(1.0, 1.0), vec2(127.1, 311.7))) * 43758.5453;
  a = fract(a); b = fract(b); c = fract(c); d = fract(d);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  vec2 shift = vec2(100.0);
  mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
  for (int i = 0; i < 6; i++) {
    v += a * noise(p);
    p = rot * p * 2.0 + shift;
    a *= 0.5;
  }
  return v;
}

void main() {
  vec2 uv = vUv;
  float t = uTime * 0.2;

  // Sky gradient (dark to deep blue)
  vec3 skyBottom = vec3(0.01, 0.01, 0.03);
  vec3 skyTop = vec3(0.02, 0.05, 0.15);
  vec3 sky = mix(skyBottom, skyTop, uv.y);

  // Aurora layers
  vec3 aurora = vec3(0.0);

  for (int i = 0; i < 4; i++) {
    float fi = float(i);
    float yOffset = 0.4 + fi * 0.1;
    float scale = 3.0 + fi * 0.5;

    vec2 p = vec2(uv.x * scale + t * (0.5 + fi * 0.2), uv.y * 2.0);
    p.x += uMouse.x * 0.5;
    float n = fbm(p + fi * 1.3);
    float n2 = fbm(p * 1.5 + fi * 2.7 + t * 0.3);

    // Vertical curtain shape
    float curtain = exp(-pow((uv.y - yOffset - n * 0.15) * 4.0, 2.0));
    curtain *= smoothstep(0.2, 0.5, uv.y) * smoothstep(0.95, 0.7, uv.y);

    float intensity = curtain * n2 * (0.8 + fi * 0.1);

    vec3 green = vec3(0.1, 0.9, 0.3);
    vec3 teal = vec3(0.0, 0.6, 0.8);
    vec3 purple = vec3(0.5, 0.1, 0.8);
    vec3 pink = vec3(0.8, 0.2, 0.5);

    vec3 layerCol;
    if (i == 0) layerCol = mix(green, teal, n);
    else if (i == 1) layerCol = mix(teal, purple, n);
    else if (i == 2) layerCol = mix(purple, pink, n);
    else layerCol = mix(green, purple, n);

    aurora += layerCol * intensity;
  }

  // Stars
  vec2 starUv = uv * vec2(uResolution.x / uResolution.y, 1.0) * 40.0;
  float star = noise(floor(starUv));
  star = step(0.98, star) * star;
  float twinkle = sin(star * 400.0 + t * 3.0) * 0.5 + 0.5;
  vec3 stars = vec3(star * twinkle * 0.8);

  vec3 col = sky + aurora * 1.5 + stars * (1.0 - length(aurora) * 0.5);

  gl_FragColor = vec4(col, 1.0);
}
`,

  // 6 - Liquid Metal
  FRAG_PREFIX + `
float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 5; i++) {
    v += a * noise(p);
    p *= 2.0;
    a *= 0.5;
  }
  return v;
}

void main() {
  vec2 uv = vUv;
  float aspect = uResolution.x / uResolution.y;
  vec2 p = (uv - 0.5) * vec2(aspect, 1.0);
  float t = uTime * 0.4;

  // Distorted reflection coordinates
  vec2 distort = vec2(
    fbm(p * 3.0 + vec2(t, 0.0)),
    fbm(p * 3.0 + vec2(0.0, t))
  );
  vec2 distort2 = vec2(
    fbm(p * 3.0 + distort * 2.0 + vec2(t * 0.7, t * 0.3)),
    fbm(p * 3.0 + distort * 2.0 + vec2(t * 0.3, t * 0.7))
  );

  // Chrome-like reflections
  float n = fbm(p * 2.0 + distort2 * 3.0);

  // Surface normal approximation
  float eps = 0.01;
  float nx = fbm(p * 2.0 + distort2 * 3.0 + vec2(eps, 0.0)) - n;
  float ny = fbm(p * 2.0 + distort2 * 3.0 + vec2(0.0, eps)) - n;
  vec3 normal = normalize(vec3(nx, ny, eps * 2.0));

  // Fake environment reflection
  vec3 viewDir = normalize(vec3(p - (uMouse - 0.5) * 0.5, 1.0));
  vec3 reflected = reflect(viewDir, normal);

  // Gradient environment
  float envAngle = atan(reflected.y, reflected.x) * 0.318;
  float envHeight = reflected.z * 0.5 + 0.5;

  vec3 envColor = vec3(0.0);
  envColor += vec3(0.8, 0.85, 0.9) * smoothstep(0.3, 0.7, envHeight);
  envColor += vec3(0.2, 0.3, 0.5) * (1.0 - envHeight);
  envColor += vec3(0.15) * sin(envAngle * 12.0 + t) * 0.5;

  // Fresnel
  float fresnel = pow(1.0 - abs(dot(vec3(0.0, 0.0, 1.0), normal)), 3.0);
  vec3 col = mix(envColor * 0.4, envColor, fresnel + 0.5);

  // Chrome specular highlights
  float spec = pow(max(reflected.z, 0.0), 20.0);
  col += vec3(1.0, 0.95, 0.9) * spec * 0.8;

  // Warm/cool color variation
  col *= 0.8 + 0.4 * vec3(
    sin(n * 6.0 + t) * 0.5 + 0.5,
    sin(n * 6.0 + t + 1.0) * 0.3 + 0.7,
    sin(n * 6.0 + t + 2.0) * 0.5 + 0.5
  );

  gl_FragColor = vec4(col, 1.0);
}
`,

  // 7 - Neon Grid
  FRAG_PREFIX + `
void main() {
  vec2 uv = vUv;
  float t = uTime;

  // Sky gradient
  vec3 col = mix(
    vec3(0.0, 0.0, 0.02),
    vec3(0.05, 0.0, 0.15),
    uv.y
  );

  // Horizon line
  float horizon = 0.35;

  // Sun
  float sunY = horizon + 0.18;
  vec2 sunCenter = vec2(0.5, sunY);
  float sunDist = length((uv - sunCenter) * vec2(uResolution.x / uResolution.y, 1.0));
  vec3 sunColor = mix(vec3(1.0, 0.3, 0.6), vec3(1.0, 0.8, 0.2), smoothstep(0.0, 0.12, sunDist));

  // Sun body with scanlines
  float sunMask = smoothstep(0.12, 0.115, sunDist);
  float scanline = step(0.5, fract(uv.y * 60.0));
  // Clip bottom half of sun with scanline gap
  float sunClip = smoothstep(horizon + 0.01, horizon + 0.16, uv.y);
  sunMask *= sunClip;
  sunMask *= mix(1.0, scanline, smoothstep(sunY - 0.05, sunY - 0.12, uv.y));
  col = mix(col, sunColor, sunMask);

  // Sun glow
  float glow = exp(-sunDist * 6.0) * 0.6;
  col += vec3(1.0, 0.2, 0.5) * glow;

  // Grid (below horizon)
  if (uv.y < horizon) {
    // Perspective transform
    float perspective = (horizon - uv.y) / horizon;
    float z = 1.0 / (perspective + 0.01);
    float x = (uv.x - 0.5) * z * 2.0;

    // Grid lines with scrolling
    float speed = t * 2.0;
    float gridZ = fract(z * 0.3 - speed * 0.1);
    float gridX = fract(x * 0.5);

    float lineZ = smoothstep(0.02, 0.0, abs(gridZ - 0.5) - 0.48);
    float lineX = smoothstep(0.02, 0.0, abs(gridX - 0.5) - 0.48);

    float grid = max(lineZ, lineX);

    // Fade with distance
    float fade = exp(-perspective * 0.3);
    grid *= fade;

    // Grid color (neon cyan/pink)
    vec3 gridColor = mix(
      vec3(0.0, 0.8, 1.0),
      vec3(1.0, 0.0, 0.8),
      sin(z * 0.1 + t * 0.5) * 0.5 + 0.5
    );

    // Mouse moves grid color
    gridColor = mix(gridColor, vec3(0.5, 0.0, 1.0), uMouse.x * 0.3);

    col += gridColor * grid * 0.8;

    // Fog at horizon
    col = mix(col, vec3(0.1, 0.0, 0.2), smoothstep(0.1, 0.0, perspective));
  }

  // Horizontal glow line at horizon
  float horizGlow = exp(-abs(uv.y - horizon) * 40.0) * 0.5;
  col += vec3(1.0, 0.3, 0.7) * horizGlow;

  // Vignette
  float vig = 1.0 - dot((uv - 0.5) * 1.2, (uv - 0.5) * 1.2);
  col *= vig;

  gl_FragColor = vec4(col, 1.0);
}
`,
]

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

export class ShaderArtEngine {
  private canvas: HTMLCanvasElement
  private gl: WebGLRenderingContext
  private quadBuffer: WebGLBuffer
  private vertShader: WebGLShader

  private programs: WebGLProgram[] = []
  private _currentIndex = 0

  // Crossfade state
  private prevProgram: WebGLProgram | null = null
  private fadeAlpha = 1.0 // 1 = fully current, 0 = fully previous
  private fading = false

  // Uniforms cache per program
  private uniformCache = new Map<WebGLProgram, {
    uTime: WebGLUniformLocation | null
    uResolution: WebGLUniformLocation | null
    uMouse: WebGLUniformLocation | null
  }>()

  // State
  private mouse = { x: 0.5, y: 0.5 }
  private time = 0
  private animationId = 0
  private lastTime = 0
  private disposed = false
  private resizeObserver: ResizeObserver | null = null

  // Crossfade FBOs
  private fboA: { framebuffer: WebGLFramebuffer; texture: WebGLTexture } | null = null
  private fboB: { framebuffer: WebGLFramebuffer; texture: WebGLTexture } | null = null
  private blendProgram: WebGLProgram | null = null
  private blendUniforms: {
    uTexA: WebGLUniformLocation | null
    uTexB: WebGLUniformLocation | null
    uMix: WebGLUniformLocation | null
  } | null = null

  constructor(private container: HTMLDivElement) {
    this.canvas = document.createElement('canvas')
    this.canvas.style.width = '100%'
    this.canvas.style.height = '100%'
    this.canvas.style.display = 'block'
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

    // Vertex shader (shared)
    this.vertShader = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC)

    // Quad buffer
    this.quadBuffer = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW)

    // Compile all shader programs
    for (const fragSrc of FRAG_SHADERS) {
      const program = this.buildProgram(fragSrc)
      this.programs.push(program)
    }

    // Blend program for crossfade
    this.initBlendProgram()

    this.resizeCanvas()
    this.initFBOs()
    this.initListeners()
    this.lastTime = performance.now()
    this.animate()
  }

  // --- Public API ---

  get shaderCount(): number {
    return this.programs.length
  }

  get shaderNames(): string[] {
    return [...SHADER_NAMES]
  }

  get currentIndex(): number {
    return this._currentIndex
  }

  setShader(index: number) {
    if (index < 0 || index >= this.programs.length) return
    if (index === this._currentIndex && !this.fading) return

    this.prevProgram = this.programs[this._currentIndex]
    this._currentIndex = index
    this.fadeAlpha = 0.0
    this.fading = true
  }

  dispose() {
    this.disposed = true
    cancelAnimationFrame(this.animationId)
    this.removeListeners()
    if (this.resizeObserver) {
      this.resizeObserver.disconnect()
      this.resizeObserver = null
    }
    this.gl.getExtension('WEBGL_lose_context')?.loseContext()
    if (this.canvas.parentElement) {
      this.canvas.parentElement.removeChild(this.canvas)
    }
  }

  // --- Internal ---

  private buildProgram(fragSrc: string): WebGLProgram {
    const gl = this.gl
    const fragShader = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc)
    const program = gl.createProgram()!
    gl.attachShader(program, this.vertShader)
    gl.attachShader(program, fragShader)
    gl.bindAttribLocation(program, 0, 'aPosition')
    gl.linkProgram(program)
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`Program link error: ${gl.getProgramInfoLog(program)}`)
    }
    // Cache uniforms
    this.uniformCache.set(program, {
      uTime: gl.getUniformLocation(program, 'uTime'),
      uResolution: gl.getUniformLocation(program, 'uResolution'),
      uMouse: gl.getUniformLocation(program, 'uMouse'),
    })
    return program
  }

  private initBlendProgram() {
    const gl = this.gl
    const fragSrc = `
      precision highp float;
      varying vec2 vUv;
      uniform sampler2D uTexA;
      uniform sampler2D uTexB;
      uniform float uMix;
      void main() {
        vec4 a = texture2D(uTexA, vUv);
        vec4 b = texture2D(uTexB, vUv);
        gl_FragColor = mix(a, b, uMix);
      }
    `
    this.blendProgram = createProgram(gl, VERT_SRC, fragSrc)
    gl.bindAttribLocation(this.blendProgram, 0, 'aPosition')
    gl.linkProgram(this.blendProgram)
    this.blendUniforms = {
      uTexA: gl.getUniformLocation(this.blendProgram, 'uTexA'),
      uTexB: gl.getUniformLocation(this.blendProgram, 'uTexB'),
      uMix: gl.getUniformLocation(this.blendProgram, 'uMix'),
    }
  }

  private createFBO() {
    const gl = this.gl
    const texture = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.canvas.width, this.canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

    const framebuffer = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)

    return { framebuffer, texture }
  }

  private initFBOs() {
    this.fboA = this.createFBO()
    this.fboB = this.createFBO()
  }

  private resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio, 2)
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    this.canvas.width = Math.floor(w * dpr)
    this.canvas.height = Math.floor(h * dpr)
  }

  private resizeFBOs() {
    const gl = this.gl
    const w = this.canvas.width
    const h = this.canvas.height

    if (this.fboA) {
      gl.bindTexture(gl.TEXTURE_2D, this.fboA.texture)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
    }
    if (this.fboB) {
      gl.bindTexture(gl.TEXTURE_2D, this.fboB.texture)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
    }
  }

  private renderProgram(program: WebGLProgram, target: WebGLFramebuffer | null) {
    const gl = this.gl
    gl.bindFramebuffer(gl.FRAMEBUFFER, target)
    gl.viewport(0, 0, this.canvas.width, this.canvas.height)
    gl.useProgram(program)

    const u = this.uniformCache.get(program)
    if (u) {
      if (u.uTime) gl.uniform1f(u.uTime, this.time)
      if (u.uResolution) gl.uniform2f(u.uResolution, this.canvas.width, this.canvas.height)
      if (u.uMouse) gl.uniform2f(u.uMouse, this.mouse.x, this.mouse.y)
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    gl.drawArrays(gl.TRIANGLE_FAN, 0, 4)
  }

  private renderBlend(texA: WebGLTexture, texB: WebGLTexture, mix: number) {
    const gl = this.gl
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.viewport(0, 0, this.canvas.width, this.canvas.height)

    gl.useProgram(this.blendProgram!)

    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texA)
    gl.uniform1i(this.blendUniforms!.uTexA, 0)

    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, texB)
    gl.uniform1i(this.blendUniforms!.uTexB, 1)

    gl.uniform1f(this.blendUniforms!.uMix, mix)

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    gl.drawArrays(gl.TRIANGLE_FAN, 0, 4)
  }

  // --- Listeners ---

  private onMouseMove = (e: MouseEvent) => {
    const rect = this.canvas.getBoundingClientRect()
    this.mouse.x = (e.clientX - rect.left) / rect.width
    this.mouse.y = 1.0 - (e.clientY - rect.top) / rect.height
  }

  private onTouchMove = (e: TouchEvent) => {
    if (e.touches.length > 0) {
      const touch = e.touches[0]
      const rect = this.canvas.getBoundingClientRect()
      this.mouse.x = (touch.clientX - rect.left) / rect.width
      this.mouse.y = 1.0 - (touch.clientY - rect.top) / rect.height
    }
  }

  private onResize = () => {
    this.resizeCanvas()
    this.resizeFBOs()
  }

  private initListeners() {
    this.canvas.addEventListener('mousemove', this.onMouseMove)
    this.canvas.addEventListener('touchmove', this.onTouchMove, { passive: true })

    this.resizeObserver = new ResizeObserver(() => {
      if (!this.disposed) this.onResize()
    })
    this.resizeObserver.observe(this.container)
  }

  private removeListeners() {
    this.canvas.removeEventListener('mousemove', this.onMouseMove)
    this.canvas.removeEventListener('touchmove', this.onTouchMove)
  }

  // --- Animation loop ---

  private animate = () => {
    if (this.disposed) return
    this.animationId = requestAnimationFrame(this.animate)

    const now = performance.now()
    const dt = Math.min((now - this.lastTime) * 0.001, 0.05)
    this.lastTime = now
    this.time += dt

    const currentProgram = this.programs[this._currentIndex]

    if (this.fading && this.prevProgram && this.fboA && this.fboB) {
      // Render previous shader to FBO A
      this.renderProgram(this.prevProgram, this.fboA.framebuffer)
      // Render current shader to FBO B
      this.renderProgram(currentProgram, this.fboB.framebuffer)

      // Advance fade
      this.fadeAlpha = Math.min(this.fadeAlpha + dt * 2.5, 1.0)
      if (this.fadeAlpha >= 1.0) {
        this.fading = false
        this.prevProgram = null
      }

      // Blend to screen
      this.renderBlend(this.fboA.texture, this.fboB.texture, this.fadeAlpha)
    } else {
      // Direct render to screen
      this.renderProgram(currentProgram, null)
    }
  }
}
