import { FFmpeg } from '@ffmpeg/ffmpeg'
import { toBlobURL } from '@ffmpeg/util'

let ffmpeg: FFmpeg | null = null
let loadPromise: Promise<FFmpeg> | null = null

const BASE_URL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd'

export async function getFFmpeg(): Promise<FFmpeg> {
  if (ffmpeg?.loaded) {
    return ffmpeg
  }

  if (loadPromise) return loadPromise

  loadPromise = (async () => {
    const instance = new FFmpeg()

    await instance.load({
      coreURL: await toBlobURL(`${BASE_URL}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${BASE_URL}/ffmpeg-core.wasm`, 'application/wasm'),
    })

    ffmpeg = instance
    return instance
  })().catch((err) => {
    loadPromise = null
    ffmpeg = null
    throw err
  })

  return loadPromise
}

export function terminateFFmpeg() {
  if (ffmpeg) {
    ffmpeg.terminate()
    ffmpeg = null
    loadPromise = null
  }
}
