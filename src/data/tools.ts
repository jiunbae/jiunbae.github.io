export interface ToolItem {
  slug: string
  title: string
  description: string
  icon: string
  tags: string[]
}

const tools: ToolItem[] = [
  {
    slug: '/tools/image-converter/',
    title: 'Image Converter',
    description: 'Convert images between PNG, JPG, WebP, and PDF with custom dimensions, scale, background, and quality',
    icon: '\uD83D\uDDBC',
    tags: ['PNG', 'JPG', 'WebP', 'SVG', 'PDF']
  },
  {
    slug: '/tools/audio-converter/',
    title: 'Audio Converter',
    description: 'Convert audio files between MP3, WAV, OGG, and AAC with custom bitrate and sample rate',
    icon: '\uD83C\uDFB5',
    tags: ['MP3', 'WAV', 'OGG', 'AAC', 'FLAC']
  },
  {
    slug: '/tools/video-converter/',
    title: 'Video Converter',
    description: 'Convert video files between MP4, WebM, and GIF with resolution and quality controls',
    icon: '\uD83C\uDFAC',
    tags: ['MP4', 'WebM', 'GIF', 'MOV', 'AVI']
  },
]

export default tools
