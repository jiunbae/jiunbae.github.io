import { FFmpeg } from '@ffmpeg/ffmpeg'
import { toBlobURL } from '@ffmpeg/util'

let ffmpeg: FFmpeg | null = null
let loadPromise: Promise<FFmpeg> | null = null
let currentProgressHandler: (({ progress }: { progress: number }) => void) | null = null

const BASE_URL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd'

export async function getFFmpeg(
  onProgress?: (progress: number) => void
): Promise<FFmpeg> {
  if (ffmpeg && ffmpeg.loaded) {
    // Re-register progress callback on cached instance
    if (currentProgressHandler) {
      ffmpeg.off('progress', currentProgressHandler)
      currentProgressHandler = null
    }
    if (onProgress) {
      currentProgressHandler = ({ progress }) => {
        onProgress(Math.min(1, Math.max(0, progress)))
      }
      ffmpeg.on('progress', currentProgressHandler)
    }
    return ffmpeg
  }

  if (loadPromise) return loadPromise

  loadPromise = (async () => {
    const instance = new FFmpeg()

    if (onProgress) {
      currentProgressHandler = ({ progress }) => {
        onProgress(Math.min(1, Math.max(0, progress)))
      }
      instance.on('progress', currentProgressHandler)
    }

    await instance.load({
      coreURL: await toBlobURL(`${BASE_URL}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${BASE_URL}/ffmpeg-core.wasm`, 'application/wasm'),
    })

    ffmpeg = instance
    return instance
  })().catch((err) => {
    loadPromise = null
    ffmpeg = null
    currentProgressHandler = null
    throw err
  })

  return loadPromise
}

export function terminateFFmpeg() {
  if (ffmpeg) {
    ffmpeg.terminate()
    ffmpeg = null
    loadPromise = null
    currentProgressHandler = null
  }
}
