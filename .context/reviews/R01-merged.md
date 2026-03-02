# R01 — Playground Projects Merged Code Review

**Reviewer**: Claude Opus 4.6 (multi-perspective)
**Date**: 2026-02-24
**Scope**: All playground projects (11 implemented) + shared infrastructure
**Perspectives**: `[SECURITY]` `[ARCHITECTURE]` `[CODE-QUALITY]` `[PERFORMANCE]` `[FRONTEND]`

---

## Critical Findings

### C-1 `[PERFORMANCE]` No WebGL context lost/restored handlers in any project

**Files**: All WebGL-based engines (`FluidSim/fluid.ts`, `Raymarching/engine.ts`, `ShaderArt/engine.ts`, `Mandelbrot/engine.ts`) and Three.js scenes (`CyberFlowers/scene.ts`, `ParticleGalaxy/scene.ts`, `AudioVisualizer/scene.ts`, `TerrainGen/engine.ts`)

None of the 8 GPU-powered projects handle the `webglcontextlost` or `webglcontextrestored` events. When the browser forcibly reclaims GPU resources (tab backgrounding on mobile, GPU driver reset, too many contexts), the canvas goes permanently black with no recovery path and no user feedback.

```ts
// Not found anywhere in the codebase:
canvas.addEventListener('webglcontextlost', ...)
canvas.addEventListener('webglcontextrestored', ...)
```

**Impact**: Permanent black screen with no recovery on mobile browsers and GPU-constrained devices.
**Recommendation**: Add context loss handlers to all GPU engines. At minimum, show a "tap to reload" overlay. For Three.js projects, use `renderer.forceContextLoss()` awareness. For raw WebGL, implement full state restore or prompt for page reload.

---

### C-2 `[PERFORMANCE]` O(n^2) collision detection in PhysicsSandbox without spatial partitioning

**File**: `/src/views/PhysicsSandbox/engine.ts` (lines 752-766)

```ts
// Collision detection & response
for (let i = 0; i < this.bodies.length; i++) {
  for (let j = i + 1; j < this.bodies.length; j++) {
    // Broad phase: bounding sphere check
    const a = this.bodies[i]
    const b = this.bodies[j]
    const distSq = v2lenSq(v2sub(a.pos, b.pos))    // new Vec2 each pair
    const sumR = a.radius + b.radius
    if (distSq > sumR * sumR * 1.2) continue
    ...
  }
}
```

With N bodies, this performs N*(N-1)/2 pair checks per frame. At 200+ bodies (easily reachable by rapid clicking), that is 20K+ comparisons per 16ms frame. Each pair also allocates a new `Vec2` via `v2sub`. Compare with the Boids project which correctly uses spatial hashing.

**Impact**: Frame drops and jank with moderate body counts; GC pressure from Vec2 allocations in the inner loop.
**Recommendation**: Implement a spatial hash grid (as in Boids) or broad-phase sweep-and-prune. Pre-allocate scratch Vec2 objects instead of creating new ones per pair.

---

### C-3 `[PERFORMANCE]` GenerativeArt Voronoi mode: per-pixel CPU nearest-point search every frame

**File**: `/src/views/GenerativeArt/engine.ts` (lines 737-795)

```ts
// Render Voronoi via per-pixel nearest-point calculation
const scale = 4  // 4x downscale helps, but still O(pixels * points)
for (let py = 0; py < sh; py++) {
  for (let px = 0; px < sw; px++) {
    // Find closest and second-closest point
    for (let i = 0; i < this.voPoints.length; i++) {
      const dx = rx - this.voPoints[i].x
      const dy = ry - this.voPoints[i].y
      const dist = dx * dx + dy * dy
      ...
    }
  }
}
```

Even at 4x downscale, for a 1920x1080 display this is ~129K pixels * ~40 points = ~5.2M distance calculations per frame on the CPU. The `createImageData` + `putImageData` path also blocks the main thread.

**Impact**: Severe frame drops on larger screens; main thread blocking causes input lag.
**Recommendation**: Move Voronoi to a WebGL shader (GPU is ideal for per-pixel parallel work), or use a jump-flood algorithm, or render to an OffscreenCanvas in a Web Worker.

---

## High Priority

### H-1 `[PERFORMANCE]` Heavy GC pressure from Vec2 object allocations in Boids simulation loop

**File**: `/src/views/Boids/engine.ts` (lines 24-67)

Every vector operation (`v2add`, `v2sub`, `v2scale`, `v2norm`, `v2setMag`, `v2limit`) creates a new `{ x, y }` object:

```ts
function v2add(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, y: a.y + b.y }
}
```

The flocking computation for 1500+ boids calls these functions dozens of times per boid per frame (separation, alignment, cohesion calculations with neighbor iteration). At 1500 boids, this is conservatively 50K+ short-lived objects per frame (~3M/second), causing frequent minor GC pauses.

**Impact**: GC micro-stutters visible as periodic frame drops, especially on mobile.
**Recommendation**: Use a mutable Vec2 approach with pre-allocated scratch variables, or switch to parallel Float32Arrays (`xs`, `ys`) for SoA (Structure of Arrays) layout. The spatial hash grid is already well-implemented with typed arrays — extend that approach to the physics.

---

### H-2 `[PERFORMANCE]` CyberFlowers `scene.traverse` on every animation frame

**File**: `/src/views/CyberFlowers/scene.ts` (lines 608-616)

```ts
// In animate(), called 60 times/second:
this.scene.traverse(child => {
  if (child instanceof THREE.Mesh && child.material instanceof THREE.ShaderMaterial) {
    if (child.material.uniforms.uTime) {
      child.material.uniforms.uTime.value = elapsed
    }
  }
})
```

`scene.traverse` walks the entire scene graph including ground plane, particles, all flower groups with their stems/centers/petals. With many flowers planted, this becomes an O(n*m) operation per frame where n is flowers and m is meshes per flower (stem + center + petals).

**Impact**: Linear performance degradation as flowers are added; unnecessary overhead for non-shader materials.
**Recommendation**: Maintain a flat array of ShaderMaterial references that need time updates. Register materials at creation time and iterate only that array.

---

### H-3 `[PERFORMANCE]` TerrainGen creates `new THREE.Color()` per vertex on every rebuild

**File**: `/src/views/TerrainGen/engine.ts` (lines 108-133)

```ts
function getBiomeColor(height: number, moisture: number): THREE.Color {
  if (height < 0.3) {
    // Sand
    const t = height / 0.3
    return new THREE.Color(0.76 + t * 0.1, 0.69 + t * 0.05, 0.5 + t * 0.1)
  }
  // ... more branches, each returning new THREE.Color(...)
}
```

For a 256x256 grid, that is 65K `new THREE.Color()` allocations per terrain rebuild. Rebuilds happen on any parameter slider change.

**Impact**: ~65K temporary object allocations per rebuild causing noticeable GC pause and potential jank during slider interaction.
**Recommendation**: Pre-allocate a single `THREE.Color` instance and reuse it via `.setRGB()`, writing directly into the color buffer attribute.

---

### H-4 `[CODE-QUALITY]` Duplicated Vec2 utility implementations across projects

**Files**: `PhysicsSandbox/engine.ts` (lines 24-75), `Boids/engine.ts` (lines 19-67)

Both projects independently implement nearly identical Vec2 helper functions (`v2add`, `v2sub`, `v2scale`, `v2len`, `v2lenSq`, `v2norm`). Similarly, `mulberry32` PRNG and `SimplexNoise2D` are duplicated verbatim between `GenerativeArt/engine.ts` and `TerrainGen/engine.ts`.

Additionally, `compileShader` and `createProgram` WebGL helpers are duplicated across `FluidSim/fluid.ts`, `Raymarching/engine.ts`, `ShaderArt/engine.ts`, and `Mandelbrot/engine.ts`.

**Impact**: Maintenance burden; bug fixes must be applied in multiple places.
**Recommendation**: Extract shared utilities into common modules: `@/utils/vec2.ts`, `@/utils/noise.ts`, `@/utils/webgl.ts`.

---

### H-5 `[ARCHITECTURE]` ImmersiveLayout `hideTimer` ref not cleared on unmount

**File**: `/src/layouts/Layout.tsx` (lines 21-31)

```ts
const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

const showHeader = useCallback(() => {
  if (hideTimer.current) clearTimeout(hideTimer.current)
  setHeaderVisible(true)
  setHintDone(true)
}, [])

const hideHeader = useCallback(() => {
  hideTimer.current = setTimeout(() => setHeaderVisible(false), 400)
}, [])
```

There is no cleanup effect to clear `hideTimer.current` on unmount. If the user navigates away while the 400ms timer is pending, `setHeaderVisible(false)` fires on an unmounted component.

**Impact**: React "setState on unmounted component" warning (in dev mode); potential minor memory leak from timer reference.
**Recommendation**: Add a cleanup effect:
```ts
useEffect(() => {
  return () => {
    if (hideTimer.current) clearTimeout(hideTimer.current)
  }
}, [])
```

---

### H-6 `[PERFORMANCE]` ShaderArt compiles all 8 shader programs eagerly at initialization

**File**: `/src/views/ShaderArt/engine.ts`

All 8 fragment shaders (Plasma, Kaleidoscope, Metaballs, Fractal Flames, Voronoi Cells, Aurora, Liquid Metal, Neon Grid) plus the crossfade blend program are compiled and linked during construction. Shader compilation is synchronous and blocks the main thread.

**Impact**: Noticeable load time stutter (~200-500ms depending on GPU driver) before first frame; most shaders are not immediately needed.
**Recommendation**: Compile shaders lazily on first use, or use `KHR_parallel_shader_compile` extension to compile asynchronously. At minimum, compile only the initial shader + blend program eagerly.

---

### H-7 `[PERFORMANCE]` Mandelbrot unnecessary requestAnimationFrame polling loop for UI state

**File**: `/src/views/Mandelbrot/Mandelbrot.tsx` (lines 60-75)

```ts
// Poll engine state for UI updates
useEffect(() => {
  if (isSSR) return
  const poll = () => {
    const engine = engineRef.current
    if (engine) {
      const [cx, cy] = engine.center
      const z = engine.zoom
      setCenterX(cx)
      setCenterY(cy)
      setZoomLevel(z)
    }
    pollRef.current = requestAnimationFrame(poll)
  }
  pollRef.current = requestAnimationFrame(poll)
  return () => cancelAnimationFrame(pollRef.current)
}, [isSSR])
```

This continuously polls engine state via rAF (~60 calls/sec), calling three React state setters every frame even when nothing has changed. Each `setCenterX`/`setCenterY`/`setZoomLevel` triggers reconciliation checks.

**Impact**: Unnecessary CPU work; triple state updates per frame causing extra reconciliation.
**Recommendation**: Use a callback-based approach where the engine notifies the React component only on actual changes (e.g., `engine.onViewChange = (cx, cy, z) => { ... }`), or debounce the polling to only run during active interaction.

---

## Medium Priority

### M-1 `[FRONTEND]` No ARIA roles or keyboard navigation on any canvas element

**Files**: All 11 project `.tsx` files

No canvas elements have `role`, `aria-label`, or `tabindex` attributes. Slider controls in panels work with keyboard (native `<input type="range">`), but there is no way to access or interact with the canvas visualizations via keyboard alone.

**Impact**: Screen reader users receive no description of the visualization; keyboard-only users cannot interact with canvas content.
**Recommendation**: Add `role="img"` and `aria-label` describing the visualization to canvas container divs. For interactive canvases (click-to-add, drag), add `tabindex="0"` and keyboard event handling.

---

### M-2 `[FRONTEND]` Hardcoded dark theme colors — no light mode support

**Files**: All `.module.scss` files across all projects

Panel backgrounds use `rgba(0,0,0,0.7)`, text colors like `#eee`, `#ccc`, `#999` are hardcoded. Canvas backgrounds are always dark (`#000`, `#0a0a0a`, `#050510`).

```scss
// Example from multiple projects:
.panel {
  background: rgba(0, 0, 0, 0.7);
  color: #eee;
}
```

The Playground listing page correctly uses CSS variables (`var(--gray-1)`, `var(--article-background)`), but the individual project pages do not.

**Impact**: Inconsistent appearance if the blog supports light mode; UI panels may be unreadable against a light background.
**Recommendation**: Since the visualizations require dark backgrounds, this is acceptable for the canvas. However, consider using CSS variables for panel styling to match the site's design system, or explicitly force dark mode on playground pages via a scoped class.

---

### M-3 `[ARCHITECTURE]` Data file lists 14 playground items but only 11 have implementations

**File**: `/src/data/playground.ts`

The data file defines entries for `cellular-automata`, `wave-interference`, and `sorting-lab`, but no corresponding view directories exist under `/src/views/`. These items will appear in the playground listing grid but lead to 404 pages.

**Impact**: Broken links on the playground listing page.
**Recommendation**: Either implement the missing 3 projects, or remove their entries from the data file, or add a "coming soon" state to the card component that disables the link.

---

### M-4 `[SECURITY]` AudioVisualizer requests microphone without clear user consent indication

**File**: `/src/views/AudioVisualizer/AudioVisualizer.tsx`

The microphone button calls `navigator.mediaDevices.getUserMedia({ audio: true })` which triggers a browser permission prompt. However, there is no pre-prompt explanation of why microphone access is needed, and if the user denies permission, the error is only logged to console.

```ts
async startMic() {
  // ...
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  // ...
  } catch (err) {
    console.error('Microphone access denied:', err)
    throw err
  }
}
```

**Impact**: Poor UX when permission is denied (no user-facing error message); no explanation of data usage.
**Recommendation**: Show an inline message explaining microphone usage before requesting. Catch the rejection in the React component and display a user-friendly error toast.

---

### M-5 `[CODE-QUALITY]` FluidSim uses `WEBGL_lose_context` extension in dispose instead of explicit resource cleanup

**File**: `/src/views/FluidSim/fluid.ts` (lines 383-395)

```ts
dispose() {
  this.disposed = true
  cancelAnimationFrame(this.animationId)
  this.removeListeners()
  if (this.resizeObserver) {
    this.resizeObserver.disconnect()
    this.resizeObserver = null
  }
  this.gl.getExtension('WEBGL_lose_context')?.loseContext()
  // No explicit deletion of textures, framebuffers, programs, shaders
  ...
}
```

While `loseContext()` eventually frees GPU resources, it is meant for testing, not production cleanup. It does not guarantee immediate resource release and is not supported uniformly. Individual FBOs, textures, and programs are not explicitly deleted.

**Impact**: Potential GPU memory leak if the extension is not available; non-deterministic cleanup timing.
**Recommendation**: Explicitly `gl.deleteTexture`, `gl.deleteFramebuffer`, `gl.deleteProgram`, and `gl.deleteShader` for all created resources before calling `loseContext()`.

---

### M-6 `[CODE-QUALITY]` Inconsistent resize handling — mix of `window.resize` and `ResizeObserver`

**Files**: Various engines across all projects

Some projects use `window.addEventListener('resize', ...)` (CyberFlowers, AudioVisualizer, GenerativeArt), while others use `ResizeObserver` (FluidSim, Boids, PhysicsSandbox, Mandelbrot, Raymarching, ShaderArt, TerrainGen). `ResizeObserver` is more accurate as it detects container size changes (not just window), but the inconsistency is a maintenance concern.

**Impact**: Projects using `window.resize` will not respond to container-level layout changes (e.g., sidebar toggle).
**Recommendation**: Standardize on `ResizeObserver` for all projects.

---

### M-7 `[ARCHITECTURE]` SSR guard pattern repeated identically across all 11 components

**Files**: All 11 `.tsx` files

Every component repeats the same boilerplate:

```tsx
const [isSSR, setIsSSR] = useState(true)

useEffect(() => {
  setIsSSR(false)
}, [])

// ...
if (isSSR) {
  return <div className={styles.page} />
}
```

This is a cross-cutting concern that could be abstracted.

**Impact**: Boilerplate duplication; any change to the SSR guard pattern requires editing all 11 files.
**Recommendation**: Create a shared `useClientOnly()` hook or a `ClientOnly` wrapper component that handles this pattern once.

---

### M-8 `[CODE-QUALITY]` PhysicsSandbox polls body count via `setInterval` instead of event-driven updates

**File**: `/src/views/PhysicsSandbox/PhysicsSandbox.tsx`

The component uses `setInterval` to periodically read `engineRef.current.bodyCount` and update React state. Similar to the Mandelbrot polling issue (H-7), but uses `setInterval` instead of rAF.

**Impact**: Unnecessary periodic wakeups; state setter called even when count has not changed.
**Recommendation**: Have the engine emit a callback when body count changes, or use a ref-based approach that only updates on meaningful changes.

---

## Low Priority

### L-1 `[CODE-QUALITY]` Non-null assertions on WebGL `createShader`/`createProgram` without fallback

**Files**: `FluidSim/fluid.ts`, `Raymarching/engine.ts`, `ShaderArt/engine.ts`, `Mandelbrot/engine.ts`

```ts
const shader = gl.createShader(type)!  // Can return null if context is lost
const program = gl.createProgram()!     // Same
```

While compilation errors are properly checked and thrown, the initial creation calls use `!` to assert non-null. If the context is in a lost state, these return null and the `!` assertion leads to a cryptic error on the next line.

**Impact**: Confusing error messages if context is lost; relates to C-1.
**Recommendation**: Add explicit null checks: `if (!shader) throw new Error('Failed to create shader — context may be lost')`.

---

### L-2 `[FRONTEND]` Glassmorphism panels may be unreadable on some backdrop-filter-unsupported browsers

**Files**: Multiple `.module.scss` files

```scss
.panel {
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}
```

Without backdrop-filter support, the panel relies on `rgba(0,0,0,0.7)` alone which is generally readable but less visually polished. The fallback is acceptable.

**Impact**: Minor visual degradation on older browsers; functionality unaffected.
**Recommendation**: Acceptable as-is. Consider `@supports (backdrop-filter: blur(1px))` for enhanced fallback if needed.

---

### L-3 `[FRONTEND]` Touch interaction could be improved on immersive pages

**File**: `/src/layouts/Layout.tsx` (lines 49-66)

The swipe-down gesture to reveal the header requires starting within the top 30px of the screen, which conflicts with some mobile browser URL bar interactions.

**Impact**: Difficulty accessing navigation on some mobile devices.
**Recommendation**: Increase the touch start zone or add a visible floating nav button on mobile.

---

### L-4 `[ARCHITECTURE]` Playground listing page has no lazy loading for card thumbnails

**File**: `/src/views/Playground/Playground.tsx`

Cards use CSS gradient backgrounds (not images), so there is no image lazy-loading concern. However, if thumbnails were added later (e.g., preview screenshots), the grid has no intersection observer setup.

**Impact**: None currently; future concern only.
**Recommendation**: Note for future enhancement if card images are added.

---

### L-5 `[CODE-QUALITY]` AudioVisualizer creates multiple `AudioContext` instances if `ensureAudioContext` is called after close

**File**: `/src/views/AudioVisualizer/scene.ts` (lines 122-126)

```ts
private ensureAudioContext() {
  if (!this.audioContext || this.audioContext.state === 'closed') {
    this.audioContext = new AudioContext()
  }
  // ...
}
```

If `dispose()` is called (which closes the AudioContext) and then somehow `ensureAudioContext` is called again (e.g., via a stale reference), a new AudioContext is created. Browsers limit the number of AudioContext instances.

**Impact**: Unlikely in practice due to the `disposed` flag, but defensive coding would prevent it.
**Recommendation**: Guard with `if (this.disposed) return` at the top of `ensureAudioContext()`.

---

### L-6 `[CODE-QUALITY]` Several projects use `window.devicePixelRatio` without `matchMedia` listener for DPR changes

**Files**: Most engine constructors

DPR can change when a user drags a window between displays with different scaling, or when browser zoom changes. Most projects read DPR once at init and on resize, but do not listen for DPR changes via `matchMedia('(resolution: ...)').

**Impact**: Canvas may appear blurry or too sharp after DPR change until next resize event.
**Recommendation**: Low priority since resize events typically accompany DPR changes. Could add `matchMedia` listener if pixel-perfect rendering is important.

---

## Per-Project Health Summary

| Project | Risk | Perf | A11y | Cleanup | Notes |
|---------|------|------|------|---------|-------|
| CyberFlowers | Low | **Med** | Low | Good | scene.traverse every frame (H-2); proper dispose with geometry/material cleanup |
| ParticleGalaxy | Low | Good | Low | Good | 80K GPU particles well-optimized; proper Three.js dispose |
| AudioVisualizer | Low | Good | Low | **Good** | Proper AudioContext.close + MediaStream.stop; no mic consent UX (M-4) |
| FluidSim | Low | Good | Low | **Med** | GPU pipeline well-optimized; incomplete explicit resource cleanup (M-5) |
| GenerativeArt | Low | **Critical** | Low | Good | Voronoi per-pixel CPU loop (C-3); other modes fine |
| Raymarching | Low | Good | Low | Good | DPR capped at 1.5; clean WebGL lifecycle |
| PhysicsSandbox | Low | **Critical** | Low | Good | O(n^2) collision (C-2); Vec2 GC pressure |
| ShaderArt | Low | **Med** | Low | Good | 8 shaders compiled eagerly (H-6); crossfade is clever |
| TerrainGen | Low | **Med** | Low | Good | Color allocation per vertex (H-3); shadow mapping well-configured |
| Mandelbrot | Low | Good | Low | Good | On-demand rendering (needsRender flag); but rAF polling in React (H-7) |
| Boids | Low | **Med** | Low | Good | Good spatial hash; but Vec2 GC pressure (H-1) |

**Risk legend**: Security exposure level
**Perf legend**: Performance under typical usage
**A11y legend**: Accessibility compliance level
**Cleanup legend**: Resource disposal completeness

---

## Statistics

| Metric | Count |
|--------|-------|
| Total files reviewed | 38 (5 shared + 33 project files) |
| Projects reviewed | 11 of 14 listed (3 unimplemented) |
| Critical findings | 3 |
| High priority findings | 7 |
| Medium priority findings | 8 |
| Low priority findings | 6 |
| Total findings | 24 |

### Findings by Perspective

| Perspective | Count |
|-------------|-------|
| PERFORMANCE | 10 |
| CODE-QUALITY | 6 |
| ARCHITECTURE | 3 |
| FRONTEND | 3 |
| SECURITY | 1 |

### Common Patterns (Positive)

1. **Consistent SSR safety** — All 11 projects properly guard against server-side rendering with the `isSSR` state pattern and dynamic imports for engines.
2. **Proper animation cleanup** — All projects use `cancelAnimationFrame` in dispose and check a `disposed` flag at the top of animation loops.
3. **Dynamic imports for code splitting** — Heavy engine modules are loaded via `import()` ensuring they are not included in the main bundle.
4. **DPR capping** — All projects cap `devicePixelRatio` at 2 (or 1.5 for Raymarching) to prevent excessive GPU load on high-DPI displays.
5. **Cancelled flag pattern** — Dynamic import cleanup correctly uses a `cancelled` boolean to prevent initialization after unmount (safe due to JS single-threaded execution model).
6. **ResizeObserver adoption** — 7 of 11 projects use `ResizeObserver` for container-aware resizing.
7. **Good Three.js disposal** — Three.js projects properly traverse scenes to dispose geometries and materials, dispose renderers and composers, and remove DOM elements.

### Common Patterns (Negative)

1. **No WebGL context loss handling** — 0 of 8 GPU projects handle context lost events.
2. **No canvas accessibility** — 0 of 11 projects have ARIA attributes on canvas elements.
3. **Immutable Vec2 in hot loops** — Both Canvas 2D physics projects (Boids, PhysicsSandbox) create thousands of short-lived Vec2 objects per frame.
4. **Code duplication** — Vec2 utils, noise functions, WebGL helpers, SSR guard patterns are all duplicated rather than shared.
5. **Polling for React state sync** — Multiple projects (Mandelbrot, PhysicsSandbox) continuously poll engine state rather than using callbacks.

---

## Actionable Summary (Priority Order)

1. **Add WebGL context loss handlers** to all 8 GPU projects (C-1) — prevents permanent black screen on mobile
2. **Add spatial partitioning** to PhysicsSandbox (C-2) — enables the project to scale beyond ~100 bodies
3. **Move Voronoi rendering to GPU** or Web Worker (C-3) — eliminates main thread blocking
4. **Pre-allocate scratch Vec2 objects** in Boids and PhysicsSandbox hot loops (H-1, part of C-2) — reduces GC pauses
5. **Cache ShaderMaterial references** in CyberFlowers instead of traversing scene (H-2) — O(1) instead of O(n)
6. **Reuse `THREE.Color` instance** in TerrainGen's `getBiomeColor` (H-3) — eliminates 65K allocations per rebuild
7. **Extract shared utilities** (Vec2, noise, WebGL helpers, SSR hook) into common modules (H-4, M-7) — reduces duplication
8. **Clear hideTimer on unmount** in ImmersiveLayout (H-5) — prevents setState on unmounted component
9. **Lazy-compile shaders** in ShaderArt (H-6) — faster initial load
10. **Replace polling with callbacks** for React state sync (H-7, M-8) — eliminates unnecessary rAF/setInterval overhead
