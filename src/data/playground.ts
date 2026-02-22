export interface PlaygroundItem {
  slug: string
  title: string
  description: string
  gradient: string
  date: string
  tags: string[]
}

const playground: PlaygroundItem[] = [
  {
    slug: '/cyber-flowers/',
    title: '사이버 꽃꽂이',
    description: 'Interactive 3D cyber flower arrangement with neon, holographic, and retro-futuristic styles',
    gradient: 'linear-gradient(135deg, #0a0a0a 0%, #001a1a 30%, #1a002a 70%, #0a0a0a 100%)',
    date: '2026-02-22',
    tags: ['Three.js', '3D', 'Interactive']
  },
  {
    slug: '/particle-galaxy/',
    title: 'Particle Galaxy',
    description: '100K+ GPU particles forming spiral galaxies — click to create gravitational wells, zoom through space',
    gradient: 'linear-gradient(135deg, #0a001a 0%, #1a0033 30%, #000d1a 70%, #0a0a2e 100%)',
    date: '2026-02-22',
    tags: ['Three.js', 'GPU', 'Simulation']
  },
  {
    slug: '/audio-visualizer/',
    title: 'Audio Visualizer',
    description: 'Real-time 3D audio visualization — drop music or use your mic, watch sound come alive',
    gradient: 'linear-gradient(135deg, #1a0000 0%, #330011 30%, #1a0033 70%, #0a001a 100%)',
    date: '2026-02-22',
    tags: ['Web Audio', 'Three.js', 'Real-time']
  },
  {
    slug: '/fluid-sim/',
    title: 'Fluid Simulation',
    description: 'GPU-accelerated fluid dynamics — paint with colorful ink, watch vortices and smoke swirl',
    gradient: 'linear-gradient(135deg, #001a0d 0%, #00331a 30%, #001a33 70%, #0a0a1a 100%)',
    date: '2026-02-22',
    tags: ['WebGL', 'GPU', 'Physics']
  },
  {
    slug: '/generative-art/',
    title: 'Generative Art Studio',
    description: 'Algorithmic art generator — flow fields, fractals, circle packing. Every click is unique',
    gradient: 'linear-gradient(135deg, #1a1a00 0%, #332200 30%, #1a0022 70%, #001a1a 100%)',
    date: '2026-02-22',
    tags: ['Canvas', 'Algorithm', 'Generative']
  },
  {
    slug: '/raymarching/',
    title: 'Raymarching Worlds',
    description: 'GPU raymarched alien landscapes — fly through procedural planets, crystals, and fractal terrains',
    gradient: 'linear-gradient(135deg, #0d001a 0%, #1a0d33 30%, #330d1a 70%, #1a0000 100%)',
    date: '2026-02-22',
    tags: ['WebGL', 'GLSL', 'Raymarching']
  },
  {
    slug: '/physics-sandbox/',
    title: 'Physics Sandbox',
    description: '2D physics playground — spawn shapes, watch them collide, connect springs, trigger explosions',
    gradient: 'linear-gradient(135deg, #1a1a0a 0%, #33220a 30%, #1a330a 70%, #0a1a0a 100%)',
    date: '2026-02-22',
    tags: ['Canvas', 'Physics', 'Interactive']
  },
  {
    slug: '/shader-art/',
    title: 'Shader Art Gallery',
    description: 'Mesmerizing GLSL shader art — plasma, kaleidoscopes, metaballs, fractals. Move your mouse to interact',
    gradient: 'linear-gradient(135deg, #1a000d 0%, #33001a 30%, #0d0033 70%, #001a1a 100%)',
    date: '2026-02-22',
    tags: ['WebGL', 'GLSL', 'Art']
  },
  {
    slug: '/terrain-gen/',
    title: 'Terrain Generator',
    description: 'Procedural 3D terrain with hydraulic erosion — adjust parameters, watch rivers form, fly through',
    gradient: 'linear-gradient(135deg, #0a1a0a 0%, #1a330d 30%, #0d1a00 70%, #001a0d 100%)',
    date: '2026-02-22',
    tags: ['Three.js', 'Procedural', 'Terrain']
  },
  {
    slug: '/mandelbrot/',
    title: 'Mandelbrot Explorer',
    description: 'Deep-zoom into the infinite fractal — smooth coloring, click to zoom, explore the boundary',
    gradient: 'linear-gradient(135deg, #00001a 0%, #000d33 30%, #1a001a 70%, #0a000a 100%)',
    date: '2026-02-22',
    tags: ['WebGL', 'GLSL', 'Fractal']
  },
  {
    slug: '/boids/',
    title: 'Boids Flocking',
    description: 'Emergent swarm behavior — thousands of agents follow simple rules to create mesmerizing flocks',
    gradient: 'linear-gradient(135deg, #0a0a1a 0%, #0d1a33 30%, #001a2e 70%, #0a0d1a 100%)',
    date: '2026-02-22',
    tags: ['Canvas', 'Simulation', 'Emergent']
  },
  {
    slug: '/cellular-automata/',
    title: 'Cellular Automata Lab',
    description: 'Conway-style automata playground — draw cells, switch rules, and watch complex life emerge',
    gradient: 'linear-gradient(135deg, #04101f 0%, #0b2245 30%, #112a38 70%, #060b1c 100%)',
    date: '2026-02-22',
    tags: ['Canvas', 'Simulation', 'Automata']
  },
  {
    slug: '/wave-interference/',
    title: 'Wave Interference',
    description: 'Interactive wave-field simulation — place sources and sculpt standing patterns in real time',
    gradient: 'linear-gradient(135deg, #220a1c 0%, #4a1630 30%, #1f0d2a 70%, #120710 100%)',
    date: '2026-02-22',
    tags: ['Canvas', 'Physics', 'Waves']
  },
  {
    slug: '/sorting-lab/',
    title: 'Sorting Lab',
    description: 'Sorting algorithm visualizer — compare quick, bubble, selection, and insertion dynamics',
    gradient: 'linear-gradient(135deg, #05201f 0%, #0b3a3d 30%, #0b223a 70%, #041319 100%)',
    date: '2026-02-22',
    tags: ['Canvas', 'Algorithm', 'Visualization']
  }
]

export default playground
