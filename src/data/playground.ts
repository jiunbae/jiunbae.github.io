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
    gradient: [
      'radial-gradient(circle at 50% 60%, rgba(0,255,255,0.35) 0%, transparent 40%)',
      'radial-gradient(circle at 30% 40%, rgba(255,0,200,0.3) 0%, transparent 30%)',
      'radial-gradient(circle at 70% 35%, rgba(0,255,150,0.25) 0%, transparent 25%)',
      'radial-gradient(circle at 55% 75%, rgba(150,0,255,0.2) 0%, transparent 20%)',
      'radial-gradient(circle at 40% 55%, rgba(0,200,255,0.15) 0%, transparent 35%)',
      'linear-gradient(180deg, #050510 0%, #0a0a1a 100%)'
    ].join(', '),
    date: '2026-02-22',
    tags: ['Three.js', '3D', 'Interactive']
  },
  {
    slug: '/particle-galaxy/',
    title: 'Particle Galaxy',
    description: '100K+ GPU particles forming spiral galaxies — click to create gravitational wells, zoom through space',
    gradient: [
      'radial-gradient(1px 1px at 20% 30%, #fff 50%, transparent 100%)',
      'radial-gradient(1px 1px at 60% 20%, #aad 50%, transparent 100%)',
      'radial-gradient(1px 1px at 80% 60%, #fff 50%, transparent 100%)',
      'radial-gradient(1px 1px at 35% 70%, #ccf 50%, transparent 100%)',
      'radial-gradient(1px 1px at 90% 40%, #fff 50%, transparent 100%)',
      'radial-gradient(1px 1px at 15% 85%, #ddf 50%, transparent 100%)',
      'radial-gradient(1px 1px at 50% 50%, #aaf 50%, transparent 100%)',
      'radial-gradient(circle at 45% 50%, rgba(100,120,255,0.3) 0%, transparent 50%)',
      'radial-gradient(ellipse at 55% 45%, rgba(180,140,255,0.15) 0%, transparent 60%)',
      'linear-gradient(180deg, #020010 0%, #0a0520 50%, #050015 100%)'
    ].join(', '),
    date: '2026-02-22',
    tags: ['Three.js', 'GPU', 'Simulation']
  },
  {
    slug: '/audio-visualizer/',
    title: 'Audio Visualizer',
    description: 'Real-time 3D audio visualization — drop music or use your mic, watch sound come alive',
    gradient: [
      'repeating-linear-gradient(90deg, transparent 0px, transparent 6px, rgba(255,50,100,0.4) 6px, rgba(255,50,100,0.4) 8px, transparent 8px, transparent 14px)',
      'repeating-linear-gradient(90deg, transparent 0px, transparent 20px, rgba(100,50,255,0.3) 20px, rgba(100,50,255,0.3) 22px, transparent 22px, transparent 40px)',
      'linear-gradient(0deg, rgba(255,0,100,0.5) 0%, rgba(100,0,255,0.3) 40%, transparent 70%)',
      'linear-gradient(180deg, #0a0010 0%, #150020 50%, #0a0015 100%)'
    ].join(', '),
    date: '2026-02-22',
    tags: ['Web Audio', 'Three.js', 'Real-time']
  },
  {
    slug: '/fluid-sim/',
    title: 'Fluid Simulation',
    description: 'GPU-accelerated fluid dynamics — paint with colorful ink, watch vortices and smoke swirl',
    gradient: [
      'radial-gradient(ellipse at 30% 40%, rgba(0,180,255,0.5) 0%, transparent 50%)',
      'radial-gradient(ellipse at 65% 55%, rgba(255,0,150,0.4) 0%, transparent 45%)',
      'radial-gradient(ellipse at 50% 70%, rgba(0,255,100,0.3) 0%, transparent 40%)',
      'radial-gradient(ellipse at 75% 30%, rgba(255,200,0,0.25) 0%, transparent 35%)',
      'linear-gradient(180deg, #050510 0%, #0a0a18 100%)'
    ].join(', '),
    date: '2026-02-22',
    tags: ['WebGL', 'GPU', 'Physics']
  },
  {
    slug: '/generative-art/',
    title: 'Generative Art Studio',
    description: 'Algorithmic art generator — flow fields, fractals, circle packing. Every click is unique',
    gradient: [
      'conic-gradient(from 45deg at 30% 40%, rgba(255,100,50,0.3), rgba(50,200,100,0.3), rgba(50,100,255,0.3), rgba(200,50,200,0.3), rgba(255,100,50,0.3))',
      'radial-gradient(circle at 70% 60%, rgba(255,200,50,0.2) 0%, transparent 40%)',
      'repeating-linear-gradient(60deg, transparent 0px, transparent 30px, rgba(255,255,255,0.03) 30px, rgba(255,255,255,0.03) 31px)',
      'linear-gradient(135deg, #0f0a15 0%, #150a10 50%, #0a1015 100%)'
    ].join(', '),
    date: '2026-02-22',
    tags: ['Canvas', 'Algorithm', 'Generative']
  },
  {
    slug: '/raymarching/',
    title: 'Raymarching Worlds',
    description: 'GPU raymarched alien landscapes — fly through procedural planets, crystals, and fractal terrains',
    gradient: [
      'radial-gradient(circle at 50% 50%, rgba(200,100,255,0.5) 0%, rgba(100,50,200,0.3) 25%, transparent 50%)',
      'radial-gradient(circle at 50% 50%, rgba(255,150,100,0.15) 20%, transparent 45%)',
      'linear-gradient(0deg, rgba(80,40,120,0.6) 0%, transparent 40%)',
      'linear-gradient(180deg, #08050f 0%, #120a1a 50%, #0a0510 100%)'
    ].join(', '),
    date: '2026-02-22',
    tags: ['WebGL', 'GLSL', 'Raymarching']
  },
  {
    slug: '/physics-sandbox/',
    title: 'Physics Sandbox',
    description: '2D physics playground — spawn shapes, watch them collide, connect springs, trigger explosions',
    gradient: [
      'radial-gradient(circle at 25% 65%, rgba(255,180,50,0.4) 0%, rgba(255,180,50,0.4) 12px, transparent 13px)',
      'radial-gradient(circle at 60% 40%, rgba(50,180,255,0.35) 0%, rgba(50,180,255,0.35) 18px, transparent 19px)',
      'radial-gradient(circle at 75% 70%, rgba(255,80,80,0.3) 0%, rgba(255,80,80,0.3) 10px, transparent 11px)',
      'radial-gradient(circle at 40% 30%, rgba(80,255,150,0.3) 0%, rgba(80,255,150,0.3) 15px, transparent 16px)',
      'linear-gradient(180deg, #0a0f15 0%, #0f1520 100%)'
    ].join(', '),
    date: '2026-02-22',
    tags: ['Canvas', 'Physics', 'Interactive']
  },
  {
    slug: '/shader-art/',
    title: 'Shader Art Gallery',
    description: 'Mesmerizing GLSL shader art — plasma, kaleidoscopes, metaballs, fractals. Move your mouse to interact',
    gradient: [
      'conic-gradient(from 0deg at 50% 50%, #ff006640, #00ff8840, #0066ff40, #ff00ff40, #ffaa0040, #00ffff40, #ff006640)',
      'radial-gradient(circle at 50% 50%, transparent 30%, rgba(0,0,0,0.5) 70%)',
      'linear-gradient(180deg, #0a0510 0%, #100818 50%, #050510 100%)'
    ].join(', '),
    date: '2026-02-22',
    tags: ['WebGL', 'GLSL', 'Art']
  },
  {
    slug: '/terrain-gen/',
    title: 'Terrain Generator',
    description: 'Procedural 3D terrain with hydraulic erosion — adjust parameters, watch rivers form, fly through',
    gradient: [
      'linear-gradient(180deg, transparent 0%, transparent 30%, rgba(60,120,60,0.15) 30%, rgba(60,120,60,0.3) 50%, rgba(80,140,80,0.4) 65%, rgba(60,100,60,0.5) 80%, rgba(40,80,40,0.6) 100%)',
      'linear-gradient(100deg, transparent 0%, transparent 20%, rgba(100,160,100,0.1) 25%, transparent 30%, transparent 45%, rgba(80,140,80,0.15) 50%, transparent 55%, transparent 70%, rgba(100,160,100,0.1) 75%, transparent 80%)',
      'linear-gradient(180deg, #0a1520 0%, #0f1a15 40%, #0a150a 100%)'
    ].join(', '),
    date: '2026-02-22',
    tags: ['Three.js', 'Procedural', 'Terrain']
  },
  {
    slug: '/mandelbrot/',
    title: 'Mandelbrot Explorer',
    description: 'Deep-zoom into the infinite fractal — smooth coloring, click to zoom, explore the boundary',
    gradient: [
      'radial-gradient(circle at 40% 50%, #000 0%, #000 15%, rgba(0,30,80,0.8) 20%, rgba(0,80,200,0.6) 25%, rgba(0,150,255,0.4) 30%, rgba(100,200,255,0.2) 35%, transparent 45%)',
      'radial-gradient(circle at 40% 50%, rgba(255,200,50,0.15) 10%, transparent 20%)',
      'conic-gradient(from 180deg at 40% 50%, rgba(0,50,150,0.2), rgba(100,0,150,0.2), rgba(0,100,200,0.2), rgba(0,50,150,0.2))',
      'linear-gradient(180deg, #000510 0%, #000a1a 50%, #000510 100%)'
    ].join(', '),
    date: '2026-02-22',
    tags: ['WebGL', 'GLSL', 'Fractal']
  },
  {
    slug: '/boids/',
    title: 'Boids Flocking',
    description: 'Emergent swarm behavior — thousands of agents follow simple rules to create mesmerizing flocks',
    gradient: [
      'radial-gradient(2px 2px at 30% 35%, rgba(100,200,255,0.8) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 35% 40%, rgba(80,180,255,0.7) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 33% 45%, rgba(120,200,255,0.6) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 40% 38%, rgba(100,210,255,0.7) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 38% 50%, rgba(90,190,255,0.6) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 45% 42%, rgba(110,200,255,0.5) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 42% 55%, rgba(80,180,255,0.5) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 55% 48%, rgba(100,200,255,0.4) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 60% 55%, rgba(120,210,255,0.4) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 65% 50%, rgba(100,200,255,0.3) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 70% 58%, rgba(80,190,255,0.3) 50%, transparent 100%)',
      'linear-gradient(180deg, #050a15 0%, #0a1020 50%, #050810 100%)'
    ].join(', '),
    date: '2026-02-22',
    tags: ['Canvas', 'Simulation', 'Emergent']
  },
  {
    slug: '/sorting-lab/',
    title: 'Sorting Lab',
    description: 'Visualize sorting algorithms in action — bubble, quick, merge, heap sort with step-by-step animation',
    gradient: [
      'repeating-linear-gradient(90deg, rgba(50,200,100,0.4) 0px, rgba(50,200,100,0.4) 4px, transparent 4px, transparent 8px)',
      'repeating-linear-gradient(90deg, rgba(255,100,50,0.3) 0px, rgba(255,100,50,0.3) 4px, transparent 4px, transparent 12px)',
      'linear-gradient(0deg, rgba(50,150,255,0.3) 0%, transparent 60%)',
      'linear-gradient(180deg, #0a0f15 0%, #0f1520 100%)'
    ].join(', '),
    date: '2026-02-22',
    tags: ['Canvas', 'Algorithm', 'Visualization']
  },
  {
    slug: '/wave-interference/',
    title: 'Wave Interference',
    description: 'Interactive wave physics — place sources, observe constructive and destructive interference patterns',
    gradient: [
      'radial-gradient(circle at 35% 50%, rgba(0,150,255,0.4) 0%, transparent 30%)',
      'radial-gradient(circle at 65% 50%, rgba(0,150,255,0.4) 0%, transparent 30%)',
      'repeating-radial-gradient(circle at 35% 50%, transparent 0px, transparent 15px, rgba(100,200,255,0.1) 16px, transparent 17px)',
      'repeating-radial-gradient(circle at 65% 50%, transparent 0px, transparent 15px, rgba(100,200,255,0.1) 16px, transparent 17px)',
      'linear-gradient(180deg, #050a18 0%, #0a1025 100%)'
    ].join(', '),
    date: '2026-02-22',
    tags: ['Canvas', 'Physics', 'Wave']
  },
  {
    slug: '/cellular-automata/',
    title: 'Cellular Automata',
    description: "Conway's Game of Life and more — watch emergent patterns evolve from simple rules",
    gradient: [
      'radial-gradient(2px 2px at 30% 40%, rgba(0,255,100,0.8) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 32% 42%, rgba(0,255,100,0.8) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 34% 40%, rgba(0,255,100,0.8) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 50% 50%, rgba(0,255,100,0.6) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 52% 48%, rgba(0,255,100,0.6) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 48% 52%, rgba(0,255,100,0.6) 50%, transparent 100%)',
      'radial-gradient(2px 2px at 70% 60%, rgba(0,255,100,0.4) 50%, transparent 100%)',
      'linear-gradient(180deg, #050a05 0%, #0a150a 50%, #050a05 100%)'
    ].join(', '),
    date: '2026-02-22',
    tags: ['Canvas', 'Simulation', 'Cellular']
  },
]

export default playground
