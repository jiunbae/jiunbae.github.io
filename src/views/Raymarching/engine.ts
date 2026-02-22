/**
 * Pure WebGL raymarching engine with 4 world presets.
 * Full-screen fragment shader with sphere tracing, Phong lighting, and AO.
 */

// ── Shader helpers ──────────────────────────────────────────────────

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

// ── Shader sources ──────────────────────────────────────────────────

const VERT_SRC = `
  attribute vec2 aPosition;
  void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
  }
`

const FRAG_SRC = `
  precision highp float;

  uniform float uTime;
  uniform vec2 uResolution;
  uniform vec2 uMouse;
  uniform int uScene;

  // ── SDF primitives ─────────────────────────────────────────────

  float sdSphere(vec3 p, float r) {
    return length(p) - r;
  }

  float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
  }

  float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
  }

  float sdPlane(vec3 p, float h) {
    return p.y - h;
  }

  float sdCappedCylinder(vec3 p, float h, float r) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
  }

  // ── SDF operations ─────────────────────────────────────────────

  float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
  }

  float opSmoothSubtraction(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
  }

  vec3 opRepeat(vec3 p, vec3 c) {
    return mod(p + 0.5 * c, c) - 0.5 * c;
  }

  mat2 rot2D(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
  }

  // ── Scene 0: Alien Planet ──────────────────────────────────────

  float sceneAlienPlanet(vec3 p) {
    // Ground plane with organic displacement
    float ground = p.y + 1.0;
    ground += sin(p.x * 0.8) * cos(p.z * 0.8) * 0.3;
    ground += sin(p.x * 2.0 + uTime * 0.3) * cos(p.z * 1.5) * 0.1;

    // Twisted columns
    float columns = 1e10;
    for (int i = 0; i < 5; i++) {
      float fi = float(i);
      float angle = fi * 1.2566; // 2PI/5
      float radius = 3.0;
      vec3 colPos = p - vec3(cos(angle) * radius, 0.0, sin(angle) * radius);

      // Twist
      float twist = sin(colPos.y * 0.5 + uTime * 0.2 + fi) * 0.5;
      colPos.xz *= rot2D(twist);

      // Organic column shape: cylinder with distortion
      float col = sdCappedCylinder(colPos, 3.0, 0.3 + sin(colPos.y * 1.5 + fi) * 0.1);
      col += sin(colPos.y * 3.0 + uTime * 0.5) * 0.05;
      col += sin(p.x * 2.0 + fi) * cos(p.z * 2.0) * 0.04;

      columns = min(columns, col);
    }

    // Floating organic orbs
    float orbs = 1e10;
    for (int i = 0; i < 3; i++) {
      float fi = float(i);
      float t = uTime * 0.3 + fi * 2.094;
      vec3 orbPos = vec3(sin(t) * 2.0, 1.5 + sin(t * 0.7 + fi) * 0.5, cos(t) * 2.0);
      float orb = sdSphere(p - orbPos, 0.25 + sin(uTime + fi) * 0.05);
      orbs = min(orbs, orb);
    }

    float scene = min(ground, columns);
    scene = min(scene, orbs);
    return scene;
  }

  // ── Scene 1: Crystal Cave ──────────────────────────────────────

  float sceneCrystalCave(vec3 p) {
    // Cavern shell (inverted sphere)
    float cavern = -(length(p) - 8.0);
    cavern += sin(p.x * 1.5) * sin(p.y * 1.5) * sin(p.z * 1.5) * 0.4;

    // Repeated crystal formations
    vec3 rep = opRepeat(p, vec3(3.0, 4.0, 3.0));

    // Rotate crystals
    float t = uTime * 0.15;
    rep.xy *= rot2D(t);
    rep.yz *= rot2D(t * 0.7);

    float crystals = sdBox(rep, vec3(0.15, 0.6, 0.15));
    crystals = min(crystals, sdBox(rep, vec3(0.6, 0.15, 0.15)));
    crystals = min(crystals, sdBox(rep, vec3(0.15, 0.15, 0.6)));

    // Stalactites from ceiling
    vec3 stalPos = p;
    stalPos.xz = mod(stalPos.xz + 1.5, 3.0) - 1.5;
    float stalactites = sdCappedCylinder(stalPos - vec3(0.0, 5.0, 0.0), 2.0, 0.1);
    stalactites -= sin(stalPos.y * 4.0) * 0.03;

    // Stalagmites from floor
    float stalagmites = sdCappedCylinder(stalPos + vec3(0.0, 5.0, 0.0), 1.5, 0.12);
    stalagmites -= sin(stalPos.y * 3.0) * 0.04;

    float scene = cavern;
    scene = min(scene, crystals);
    scene = min(scene, stalactites);
    scene = min(scene, stalagmites);
    return scene;
  }

  // ── Scene 2: Fractal Landscape ─────────────────────────────────

  float sceneFractalLandscape(vec3 p) {
    // Multi-octave terrain
    float terrain = p.y;

    // Mandelbulb-inspired displacement
    float scale = 1.0;
    float detail = 0.0;
    vec3 q = p * 0.3;
    for (int i = 0; i < 5; i++) {
      detail += abs(sin(q.x * scale) * cos(q.z * scale)) / scale;
      scale *= 2.17;
      q.xz *= rot2D(0.45);
    }
    terrain += detail * 1.5 - 2.0;

    // Add some sharp ridges
    float ridges = abs(sin(p.x * 0.5 + sin(p.z * 0.3)) * cos(p.z * 0.5)) * 1.2;
    terrain -= ridges;

    // Floating fractal-esque rocks
    float rocks = 1e10;
    for (int i = 0; i < 4; i++) {
      float fi = float(i);
      float t = uTime * 0.1;
      vec3 rockPos = vec3(
        sin(fi * 1.7 + t) * 5.0,
        3.0 + sin(fi * 2.3 + t * 0.5) * 1.0,
        cos(fi * 1.3 + t * 0.7) * 5.0
      );
      float rock = sdSphere(p - rockPos, 0.3 + sin(fi + uTime * 0.3) * 0.1);
      // Fractal displacement on rocks
      vec3 rp = (p - rockPos) * 4.0;
      rock += (sin(rp.x) * sin(rp.y) * sin(rp.z)) * 0.05;
      rocks = min(rocks, rock);
    }

    return min(terrain, rocks);
  }

  // ── Scene 3: Abstract Geometry ─────────────────────────────────

  float sceneAbstractGeometry(vec3 p) {
    float t = uTime * 0.3;

    // Central rotating torus
    vec3 p1 = p;
    p1.xy *= rot2D(t);
    float torus1 = sdTorus(p1, vec2(2.0, 0.3));

    // Second torus, orthogonal
    vec3 p2 = p;
    p2.yz *= rot2D(t * 0.7);
    float torus2 = sdTorus(p2, vec2(2.0, 0.3));

    // Third torus
    vec3 p3 = p;
    p3.xz *= rot2D(t * 1.1);
    float torus3 = sdTorus(p3, vec2(2.0, 0.3));

    // Smooth union of toruses
    float toruses = opSmoothUnion(torus1, torus2, 0.5);
    toruses = opSmoothUnion(toruses, torus3, 0.5);

    // Central sphere
    float sphere = sdSphere(p, 1.0 + sin(t) * 0.2);

    // Orbiting smaller spheres
    float orbitals = 1e10;
    for (int i = 0; i < 6; i++) {
      float fi = float(i);
      float angle = fi * 1.0472 + t; // 2PI/6
      vec3 oPos = vec3(
        cos(angle) * 3.0,
        sin(fi * 0.5 + t * 0.5) * 1.0,
        sin(angle) * 3.0
      );
      float orb = sdSphere(p - oPos, 0.3);
      orbitals = min(orbitals, orb);
    }

    // Smooth union everything
    float scene = opSmoothUnion(toruses, sphere, 0.3);
    scene = opSmoothUnion(scene, orbitals, 0.4);

    return scene;
  }

  // ── Main distance function ─────────────────────────────────────

  float map(vec3 p) {
    if (uScene == 0) return sceneAlienPlanet(p);
    if (uScene == 1) return sceneCrystalCave(p);
    if (uScene == 2) return sceneFractalLandscape(p);
    return sceneAbstractGeometry(p);
  }

  // ── Normal calculation (central differences) ──────────────────

  vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
      map(p + e.xyy) - map(p - e.xyy),
      map(p + e.yxy) - map(p - e.yxy),
      map(p + e.yyx) - map(p - e.yyx)
    ));
  }

  // ── Sphere tracing ────────────────────────────────────────────

  float raymarch(vec3 ro, vec3 rd) {
    float t = 0.0;
    for (int i = 0; i < 128; i++) {
      vec3 p = ro + rd * t;
      float d = map(p);
      if (d < 0.001) break;
      t += d;
      if (t > 100.0) break;
    }
    return t;
  }

  // ── Ambient occlusion approximation ────────────────────────────

  float calcAO(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; i++) {
      float h = 0.01 + 0.12 * float(i);
      float d = map(p + h * n);
      occ += (h - d) * sca;
      sca *= 0.95;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
  }

  // ── Soft shadow ───────────────────────────────────────────────

  float softShadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float res = 1.0;
    float t = mint;
    for (int i = 0; i < 32; i++) {
      float h = map(ro + rd * t);
      res = min(res, k * h / t);
      t += clamp(h, 0.02, 0.1);
      if (h < 0.001 || t > maxt) break;
    }
    return clamp(res, 0.0, 1.0);
  }

  // ── Scene-specific coloring ────────────────────────────────────

  vec3 getSceneColor(vec3 p, vec3 n, vec3 rd, float t) {
    vec3 lightDir = normalize(vec3(0.8, 0.6, -0.5));
    float diff = max(dot(n, lightDir), 0.0);
    float spec = pow(max(dot(reflect(rd, n), lightDir), 0.0), 32.0);
    float ao = calcAO(p, n);
    float shadow = softShadow(p + n * 0.01, lightDir, 0.02, 10.0, 8.0);

    vec3 col;

    if (uScene == 0) {
      // Alien Planet - warm organic tones
      vec3 albedo = vec3(0.3, 0.5, 0.4);
      // Height-based color: lower = dark, higher = bright teal
      albedo = mix(vec3(0.15, 0.1, 0.08), vec3(0.2, 0.7, 0.5), clamp(p.y * 0.3 + 0.5, 0.0, 1.0));
      // Columns: more saturated
      if (abs(p.y) < 3.0 && length(p.xz) > 1.5 && length(p.xz) < 4.5) {
        albedo = mix(albedo, vec3(0.6, 0.2, 0.8), 0.3);
      }
      vec3 ambient = vec3(0.08, 0.05, 0.12) * ao;
      col = ambient + albedo * diff * shadow + vec3(0.6, 0.8, 0.9) * spec * 0.3;

      // Sky gradient for distant fragments
      vec3 sky = mix(vec3(0.15, 0.05, 0.2), vec3(0.4, 0.2, 0.5), clamp(rd.y * 0.5 + 0.5, 0.0, 1.0));
      float fog = 1.0 - exp(-t * 0.04);
      col = mix(col, sky, fog);

    } else if (uScene == 1) {
      // Crystal Cave - dark with colored light sources
      vec3 albedo = vec3(0.3, 0.35, 0.5);
      // Add crystalline iridescence
      float irid = sin(dot(p, vec3(1.0, 2.0, 3.0)) * 3.0 + uTime * 0.5);
      albedo += vec3(0.15, 0.1, 0.25) * irid;

      // Point lights in the cave
      vec3 lightAccum = vec3(0.0);
      for (int i = 0; i < 3; i++) {
        float fi = float(i);
        vec3 lp = vec3(sin(fi * 2.094 + uTime * 0.2) * 3.0, sin(fi + uTime * 0.3), cos(fi * 2.094 + uTime * 0.2) * 3.0);
        vec3 lCol = vec3(0.0);
        if (i == 0) lCol = vec3(0.3, 0.5, 1.0);
        if (i == 1) lCol = vec3(1.0, 0.3, 0.5);
        if (i == 2) lCol = vec3(0.3, 1.0, 0.5);
        float dist = length(p - lp);
        float atten = 1.0 / (1.0 + dist * dist * 0.1);
        float ld = max(dot(n, normalize(lp - p)), 0.0);
        lightAccum += lCol * ld * atten;
      }

      vec3 ambient = vec3(0.02, 0.02, 0.04) * ao;
      col = ambient + albedo * (diff * shadow * 0.3 + lightAccum) + vec3(0.5) * spec * 0.2;

      // Dark cave fog
      float fog = 1.0 - exp(-t * 0.08);
      col = mix(col, vec3(0.01, 0.01, 0.03), fog);

    } else if (uScene == 2) {
      // Fractal Landscape - earthy terrain tones
      vec3 albedo = vec3(0.35, 0.25, 0.15);
      // Height-based: low=green, mid=brown, high=snow
      float h = p.y;
      if (h < -0.5) albedo = vec3(0.1, 0.25, 0.1); // grass
      else if (h < 0.5) albedo = vec3(0.35, 0.25, 0.15); // dirt/rock
      else if (h < 1.5) albedo = vec3(0.4, 0.38, 0.35); // stone
      else albedo = vec3(0.85, 0.88, 0.95); // snow

      // Normal-based grass on slopes
      if (n.y > 0.7 && h < 0.5) {
        albedo = mix(albedo, vec3(0.15, 0.35, 0.1), (n.y - 0.7) * 3.0);
      }

      vec3 skyCol = mix(vec3(0.6, 0.7, 0.9), vec3(0.2, 0.3, 0.6), clamp(rd.y, 0.0, 1.0));
      vec3 ambient = vec3(0.08, 0.1, 0.15) * ao;
      col = ambient + albedo * diff * shadow + vec3(0.8, 0.85, 1.0) * spec * 0.15;

      float fog = 1.0 - exp(-t * 0.025);
      col = mix(col, skyCol * 0.5, fog);

    } else {
      // Abstract Geometry - clean modern look
      vec3 albedo = vec3(0.9, 0.92, 0.95);
      // Fresnel for rim light
      float fresnel = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);

      // Color based on normal direction
      vec3 normColor = n * 0.5 + 0.5;
      albedo = mix(albedo, normColor * 0.8 + 0.2, 0.3);

      vec3 lightDir2 = normalize(vec3(-0.5, 0.8, 0.3));
      float diff2 = max(dot(n, lightDir2), 0.0) * 0.3;

      vec3 ambient = vec3(0.12, 0.1, 0.15) * ao;
      col = ambient + albedo * (diff * shadow + diff2) + vec3(1.0) * spec * 0.5;
      col += vec3(0.3, 0.5, 1.0) * fresnel * 0.4;

      // Light fog
      vec3 bgCol = vec3(0.04, 0.04, 0.06);
      float fog = 1.0 - exp(-t * 0.03);
      col = mix(col, bgCol, fog);
    }

    return col;
  }

  // ── Main ──────────────────────────────────────────────────────

  void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;

    // Camera: mouse X controls orbit angle, mouse Y controls height
    float mouseX = uMouse.x * 2.0 - 1.0; // -1 to 1
    float mouseY = uMouse.y * 2.0 - 1.0; // -1 to 1
    float camAngle = mouseX * 3.14159;
    float camHeight = mouseY * 3.0;

    vec3 ro, lookAt;

    if (uScene == 0) {
      // Alien planet: orbit around center
      ro = vec3(cos(camAngle) * 6.0, 2.0 + camHeight, sin(camAngle) * 6.0);
      lookAt = vec3(0.0, 0.5, 0.0);
    } else if (uScene == 1) {
      // Crystal cave: orbit inside
      ro = vec3(cos(camAngle) * 3.0, camHeight * 0.5, sin(camAngle) * 3.0);
      lookAt = vec3(0.0, 0.0, 0.0);
    } else if (uScene == 2) {
      // Fractal landscape: orbit from above
      float orbitT = uTime * 0.05 + camAngle * 0.3;
      ro = vec3(cos(orbitT) * 8.0, 5.0 + camHeight, sin(orbitT) * 8.0);
      lookAt = vec3(0.0, -0.5, 0.0);
    } else {
      // Abstract: orbit around
      ro = vec3(cos(camAngle) * 5.5, 1.5 + camHeight, sin(camAngle) * 5.5);
      lookAt = vec3(0.0, 0.0, 0.0);
    }

    // Camera matrix
    vec3 forward = normalize(lookAt - ro);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(right, forward);
    vec3 rd = normalize(forward + uv.x * right + uv.y * up);

    // Raymarch
    float t = raymarch(ro, rd);

    vec3 col;

    if (t < 100.0) {
      vec3 p = ro + rd * t;
      vec3 n = calcNormal(p);
      col = getSceneColor(p, n, rd, t);
    } else {
      // Background
      if (uScene == 0) {
        // Sky gradient
        col = mix(vec3(0.15, 0.05, 0.2), vec3(0.4, 0.2, 0.5), clamp(rd.y * 0.5 + 0.5, 0.0, 1.0));
        // Stars
        vec3 starUV = rd * 200.0;
        float star = step(0.98, fract(sin(dot(floor(starUV), vec3(12.9898, 78.233, 45.164))) * 43758.5453));
        col += star * 0.4;
      } else if (uScene == 1) {
        col = vec3(0.01, 0.01, 0.03);
      } else if (uScene == 2) {
        col = mix(vec3(0.6, 0.7, 0.9), vec3(0.2, 0.3, 0.6), clamp(rd.y, 0.0, 1.0)) * 0.5;
      } else {
        col = vec3(0.04, 0.04, 0.06);
      }
    }

    // Gamma correction
    col = pow(col, vec3(0.4545));

    // Vignette
    vec2 vUv = gl_FragCoord.xy / uResolution;
    vec2 vig = vUv * (1.0 - vUv);
    col *= pow(vig.x * vig.y * 16.0, 0.15);

    gl_FragColor = vec4(col, 1.0);
  }
`

// ── Scene names ─────────────────────────────────────────────────────

const SCENE_NAMES = [
  'Alien Planet',
  'Crystal Cave',
  'Fractal Landscape',
  'Abstract Geometry'
]

// ── Engine ──────────────────────────────────────────────────────────

export class RaymarchEngine {
  private canvas: HTMLCanvasElement
  private gl: WebGLRenderingContext
  private program: WebGLProgram
  private quadBuffer: WebGLBuffer

  // Uniform locations
  private uTime: WebGLUniformLocation | null
  private uResolution: WebGLUniformLocation | null
  private uMouse: WebGLUniformLocation | null
  private uScene: WebGLUniformLocation | null

  private animationId = 0
  private disposed = false
  private startTime: number
  private mouseX = 0.5
  private mouseY = 0.5
  private scene = 0

  constructor(private container: HTMLDivElement) {
    this.canvas = document.createElement('canvas')
    this.canvas.style.display = 'block'
    this.canvas.style.width = '100%'
    this.canvas.style.height = '100%'
    container.appendChild(this.canvas)

    const gl = this.canvas.getContext('webgl', {
      alpha: false,
      depth: false,
      stencil: false,
      antialias: false,
      preserveDrawingBuffer: false
    })
    if (!gl) throw new Error('WebGL not supported')
    this.gl = gl

    this.program = createProgram(gl, VERT_SRC, FRAG_SRC)
    gl.useProgram(this.program)

    // Uniform locations
    this.uTime = gl.getUniformLocation(this.program, 'uTime')
    this.uResolution = gl.getUniformLocation(this.program, 'uResolution')
    this.uMouse = gl.getUniformLocation(this.program, 'uMouse')
    this.uScene = gl.getUniformLocation(this.program, 'uScene')

    // Full-screen quad
    this.quadBuffer = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, -1, 1, 1, 1,
      -1, -1, 1, 1, 1, -1
    ]), gl.STATIC_DRAW)

    const aPosition = gl.getAttribLocation(this.program, 'aPosition')
    gl.enableVertexAttribArray(aPosition)
    gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0)

    this.startTime = performance.now()

    this.resize()
    window.addEventListener('resize', this.handleResize)
    this.canvas.addEventListener('mousemove', this.handleMouseMove)
    this.canvas.addEventListener('touchmove', this.handleTouchMove, { passive: false })

    this.animate()
  }

  // ── Public API ────────────────────────────────────────────────

  setScene(index: number) {
    this.scene = Math.max(0, Math.min(index, SCENE_NAMES.length - 1))
  }

  get sceneCount(): number {
    return SCENE_NAMES.length
  }

  get sceneNames(): string[] {
    return [...SCENE_NAMES]
  }

  get currentScene(): number {
    return this.scene
  }

  dispose() {
    this.disposed = true
    cancelAnimationFrame(this.animationId)
    window.removeEventListener('resize', this.handleResize)
    this.canvas.removeEventListener('mousemove', this.handleMouseMove)
    this.canvas.removeEventListener('touchmove', this.handleTouchMove)
    this.gl.getExtension('WEBGL_lose_context')?.loseContext()
    if (this.canvas.parentElement) {
      this.canvas.parentElement.removeChild(this.canvas)
    }
  }

  // ── Internal ──────────────────────────────────────────────────

  private handleResize = () => {
    this.resize()
  }

  private resize() {
    const dpr = Math.min(window.devicePixelRatio, 1.5) // Limit for raymarching performance
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    this.canvas.width = Math.floor(w * dpr)
    this.canvas.height = Math.floor(h * dpr)
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height)
  }

  private handleMouseMove = (e: MouseEvent) => {
    const rect = this.canvas.getBoundingClientRect()
    this.mouseX = (e.clientX - rect.left) / rect.width
    this.mouseY = 1.0 - (e.clientY - rect.top) / rect.height
  }

  private handleTouchMove = (e: TouchEvent) => {
    e.preventDefault()
    const touch = e.touches[0]
    const rect = this.canvas.getBoundingClientRect()
    this.mouseX = (touch.clientX - rect.left) / rect.width
    this.mouseY = 1.0 - (touch.clientY - rect.top) / rect.height
  }

  private animate = () => {
    if (this.disposed) return
    this.animationId = requestAnimationFrame(this.animate)

    const gl = this.gl
    const time = (performance.now() - this.startTime) * 0.001

    gl.uniform1f(this.uTime, time)
    gl.uniform2f(this.uResolution, this.canvas.width, this.canvas.height)
    gl.uniform2f(this.uMouse, this.mouseX, this.mouseY)
    gl.uniform1i(this.uScene, this.scene)

    gl.drawArrays(gl.TRIANGLES, 0, 6)
  }
}
