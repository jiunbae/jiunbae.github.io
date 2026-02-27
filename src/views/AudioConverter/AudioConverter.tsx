import { useState, useRef, useCallback, useEffect } from 'react'
import type { HeadProps } from 'gatsby'
import { Link } from 'gatsby'
import { Seo } from '@/components'
import * as styles from './AudioConverter.module.scss'

type OutputFormat = 'mp3' | 'wav' | 'ogg' | 'aac'
type Bitrate = '128k' | '192k' | '256k' | '320k'
type SampleRate = 22050 | 44100 | 48000

const ACCEPT_TYPES = '.mp3,.wav,.ogg,.flac,.aac,.m4a,.wma,audio/*'

const formatLabel = (fmt: OutputFormat) => {
  switch (fmt) {
    case 'mp3': return 'MP3'
    case 'wav': return 'WAV'
    case 'ogg': return 'OGG'
    case 'aac': return 'AAC'
  }
}

const formatExt = (fmt: OutputFormat) => {
  if (fmt === 'aac') return 'm4a'
  return fmt
}

const formatSize = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

const getCodecArgs = (format: OutputFormat, bitrate: Bitrate, sampleRate: SampleRate): string[] => {
  const args: string[] = ['-ar', String(sampleRate)]

  switch (format) {
    case 'mp3':
      args.push('-c:a', 'libmp3lame', '-b:a', bitrate)
      break
    case 'wav':
      args.push('-c:a', 'pcm_s16le')
      break
    case 'ogg':
      args.push('-c:a', 'libvorbis', '-b:a', bitrate)
      break
    case 'aac':
      args.push('-c:a', 'aac', '-b:a', bitrate)
      break
  }

  return args
}

const AudioConverterPage = () => {
  const [isSSR, setIsSSR] = useState(true)
  const [file, setFile] = useState<File | null>(null)
  const [format, setFormat] = useState<OutputFormat>('mp3')
  const [bitrate, setBitrate] = useState<Bitrate>('192k')
  const [sampleRate, setSampleRate] = useState<SampleRate>(44100)
  const [isDragging, setIsDragging] = useState(false)
  const [isConverting, setIsConverting] = useState(false)
  const [isLoadingFFmpeg, setIsLoadingFFmpeg] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  const handleFile = useCallback((f: File) => {
    setFile(f)
    setError(null)
    setProgress(0)
  }, [])

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) handleFile(f)
  }, [handleFile])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }, [handleFile])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleRemove = useCallback(() => {
    setFile(null)
    setError(null)
    setProgress(0)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [])

  const handleConvert = useCallback(async () => {
    if (!file) return
    setIsConverting(true)
    setError(null)
    setProgress(0)

    try {
      setIsLoadingFFmpeg(true)
      const { getFFmpeg } = await import('@/utils/ffmpeg')
      const { fetchFile } = await import('@ffmpeg/util')
      const ffmpeg = await getFFmpeg((p) => setProgress(p))
      setIsLoadingFFmpeg(false)

      const ext = (file.name.split('.').pop() || 'bin').replace(/[^a-zA-Z0-9]/g, '')
      const inputName = `input.${ext}`
      const outputExt = formatExt(format)
      const outputName = `output.${outputExt}`

      await ffmpeg.writeFile(inputName, await fetchFile(file))

      const codecArgs = getCodecArgs(format, bitrate, sampleRate)
      await ffmpeg.exec(['-i', inputName, ...codecArgs, '-y', outputName])

      const data = await ffmpeg.readFile(outputName)
      const blob = new Blob([data], { type: `audio/${format}` })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const baseName = file.name.replace(/\.[^.]+$/, '')
      a.download = `${baseName}.${outputExt}`
      a.click()
      URL.revokeObjectURL(url)

      // Cleanup
      await ffmpeg.deleteFile(inputName).catch(() => {})
      await ffmpeg.deleteFile(outputName).catch(() => {})

      setProgress(1)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Conversion failed')
    } finally {
      setIsConverting(false)
      setIsLoadingFFmpeg(false)
    }
  }, [file, format, bitrate, sampleRate])

  if (isSSR) return null

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <Link to="/tools/" className={styles.backLink}>Tools</Link>
        <h1 className={styles.title}>Audio Converter</h1>
        <p className={styles.subtitle}>
          Convert audio files between MP3, WAV, OGG, and AAC — runs entirely in your browser
        </p>
      </div>

      <div className={styles.layout}>
        {/* Left: Upload + Info */}
        <div className={styles.previewSection}>
          {!file ? (
            <div
              className={`${styles.dropZone} ${isDragging ? styles.dragging : ''}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click() }}
            >
              <div className={styles.dropIcon}>
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M9 18V5l12-2v13" />
                  <circle cx="6" cy="18" r="3" />
                  <circle cx="18" cy="16" r="3" />
                </svg>
              </div>
              <p className={styles.dropText}>Drop an audio file here or click to browse</p>
              <p className={styles.dropHint}>MP3, WAV, OGG, FLAC, AAC, M4A, WMA</p>
              <input
                ref={fileInputRef}
                type="file"
                accept={ACCEPT_TYPES}
                onChange={handleFileChange}
                className={styles.hiddenInput}
              />
            </div>
          ) : (
            <div className={styles.fileInfo}>
              <div className={styles.previewToolbar}>
                <span className={styles.fileName}>{file.name}</span>
                <button className={styles.removeBtn} onClick={handleRemove}>
                  Remove
                </button>
              </div>
              <div className={styles.fileDetails}>
                <div className={styles.fileIcon}>
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M9 18V5l12-2v13" />
                    <circle cx="6" cy="18" r="3" />
                    <circle cx="18" cy="16" r="3" />
                  </svg>
                </div>
                <div className={styles.fileMeta}>
                  <span className={styles.fileSize}>{formatSize(file.size)}</span>
                  <span className={styles.fileType}>{file.type || 'audio'}</span>
                </div>
              </div>

              {(isConverting || progress > 0) && (
                <div className={styles.progressSection}>
                  <div className={styles.progressBar}>
                    <div
                      className={styles.progressFill}
                      style={{ width: `${Math.round(progress * 100)}%` }}
                    />
                  </div>
                  <span className={styles.progressText}>
                    {isLoadingFFmpeg ? 'Loading FFmpeg...' : isConverting ? `${Math.round(progress * 100)}%` : 'Done'}
                  </span>
                </div>
              )}

              {error && (
                <div className={styles.error}>{error}</div>
              )}
            </div>
          )}
        </div>

        {/* Right: Controls */}
        <div className={styles.controls}>
          <fieldset className={styles.fieldset}>
            <legend className={styles.legend}>Format</legend>
            <div className={styles.formatToggle}>
              {(['mp3', 'wav', 'ogg', 'aac'] as OutputFormat[]).map(f => (
                <button
                  key={f}
                  className={`${styles.formatBtn} ${format === f ? styles.active : ''}`}
                  onClick={() => setFormat(f)}
                >
                  {formatLabel(f)}
                </button>
              ))}
            </div>
          </fieldset>

          {format !== 'wav' && (
            <fieldset className={styles.fieldset}>
              <legend className={styles.legend}>Bitrate</legend>
              <div className={styles.formatToggle}>
                {(['128k', '192k', '256k', '320k'] as Bitrate[]).map(b => (
                  <button
                    key={b}
                    className={`${styles.formatBtn} ${bitrate === b ? styles.active : ''}`}
                    onClick={() => setBitrate(b)}
                  >
                    {b}
                  </button>
                ))}
              </div>
            </fieldset>
          )}

          <fieldset className={styles.fieldset}>
            <legend className={styles.legend}>Sample Rate</legend>
            <div className={styles.formatToggle}>
              {([22050, 44100, 48000] as SampleRate[]).map(sr => (
                <button
                  key={sr}
                  className={`${styles.formatBtn} ${sampleRate === sr ? styles.active : ''}`}
                  onClick={() => setSampleRate(sr)}
                >
                  {sr === 22050 ? '22.05k' : sr === 44100 ? '44.1k' : '48k'}
                </button>
              ))}
            </div>
          </fieldset>

          <button
            className={styles.downloadBtn}
            onClick={handleConvert}
            disabled={!file || isConverting}
          >
            {isLoadingFFmpeg ? 'Loading FFmpeg...' : isConverting ? 'Converting...' : `Convert to ${formatLabel(format)}`}
          </button>

          <p className={styles.note}>
            FFmpeg WASM loads ~25MB on first use. Conversion runs entirely in your browser.
          </p>
        </div>
      </div>
    </div>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => (
  <Seo
    title="Audio Converter"
    description="Convert audio files between MP3, WAV, OGG, and AAC — custom bitrate and sample rate. Runs entirely in your browser with FFmpeg WASM."
    heroImage=""
    pathname={pathname}
  />
)

export default AudioConverterPage
