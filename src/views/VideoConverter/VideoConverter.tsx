import { useState, useRef, useCallback, useEffect } from 'react'
import * as styles from './VideoConverter.module.scss'

type OutputFormat = 'mp4' | 'webm' | 'gif'
type Resolution = 'original' | '1080' | '720' | '480'

const ACCEPT_TYPES = '.mp4,.webm,.mov,.avi,.mkv,video/*'
const SIZE_WARNING_BYTES = 200 * 1024 * 1024 // 200MB

const formatLabel = (fmt: OutputFormat) => {
  switch (fmt) {
    case 'mp4': return 'MP4'
    case 'webm': return 'WebM'
    case 'gif': return 'GIF'
  }
}

const resolutionLabel = (res: Resolution) => {
  switch (res) {
    case 'original': return 'Original'
    case '1080': return '1080p'
    case '720': return '720p'
    case '480': return '480p'
  }
}

const formatSize = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

const getCodecArgs = (format: OutputFormat, resolution: Resolution, crf: number): string[] => {
  const args: string[] = []

  // Scale filter
  if (resolution !== 'original') {
    const h = resolution === '1080' ? 1080 : resolution === '720' ? 720 : 480
    args.push('-vf', `scale=-2:${h}`)
  }

  switch (format) {
    case 'mp4':
      args.push('-c:v', 'libx264', '-crf', String(crf), '-preset', 'fast', '-c:a', 'aac', '-b:a', '128k')
      break
    case 'webm':
      args.push('-c:v', 'libvpx', '-crf', String(crf), '-b:v', '0', '-c:a', 'libvorbis', '-b:a', '128k')
      break
    case 'gif':
      // GIF: use palette for better quality
      args.length = 0 // clear scale filter, we'll include it in the filtergraph
      if (resolution !== 'original') {
        const h = resolution === '1080' ? 1080 : resolution === '720' ? 720 : 480
        args.push('-vf', `scale=-2:${h},fps=15`)
      } else {
        args.push('-vf', 'fps=15')
      }
      args.push('-an')
      break
  }

  return args
}

const VideoConverterPage = () => {
  const [isSSR, setIsSSR] = useState(true)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [format, setFormat] = useState<OutputFormat>('mp4')
  const [resolution, setResolution] = useState<Resolution>('original')
  const [crf, setCrf] = useState(23)
  const [isDragging, setIsDragging] = useState(false)
  const [isConverting, setIsConverting] = useState(false)
  const [isLoadingFFmpeg, setIsLoadingFFmpeg] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const prevUrlRef = useRef<string | null>(null)

  useEffect(() => {
    setIsSSR(false)
  }, [])

  useEffect(() => {
    return () => {
      if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current)
    }
  }, [])

  const handleFile = useCallback((f: File) => {
    if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current)

    const url = URL.createObjectURL(f)
    prevUrlRef.current = url
    setFile(f)
    setPreviewUrl(url)
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
    if (prevUrlRef.current) {
      URL.revokeObjectURL(prevUrlRef.current)
      prevUrlRef.current = null
    }
    setFile(null)
    setPreviewUrl(null)
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

      const inputExt = (file.name.split('.').pop() || 'mp4').replace(/[^a-zA-Z0-9]/g, '')
      const inputName = `input.${inputExt}`
      const outputName = `output.${format}`

      await ffmpeg.writeFile(inputName, await fetchFile(file))

      const codecArgs = getCodecArgs(format, resolution, crf)
      await ffmpeg.exec(['-i', inputName, ...codecArgs, '-y', outputName])

      const data = await ffmpeg.readFile(outputName)
      const mimeType = format === 'gif' ? 'image/gif' : `video/${format}`
      const blob = new Blob([data], { type: mimeType })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const baseName = file.name.replace(/\.[^.]+$/, '')
      a.download = `${baseName}.${format}`
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
  }, [file, format, resolution, crf])

  if (isSSR) return null

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <a href="/tools/" className={styles.backLink}>Tools</a>
        <h1 className={styles.title}>Video Converter</h1>
        <p className={styles.subtitle}>
          Convert video files between MP4, WebM, and GIF — runs entirely in your browser
        </p>
      </div>

      <div className={styles.layout}>
        {/* Left: Upload + Preview */}
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
                  <polygon points="23 7 16 12 23 17 23 7" />
                  <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                </svg>
              </div>
              <p className={styles.dropText}>Drop a video file here or click to browse</p>
              <p className={styles.dropHint}>MP4, WebM, MOV, AVI, MKV</p>
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

              {previewUrl && (
                <div className={styles.videoPreview}>
                  <video src={previewUrl} controls muted className={styles.video} />
                </div>
              )}

              <div className={styles.fileMeta}>
                <span className={styles.fileSize}>{formatSize(file.size)}</span>
                {file.size > SIZE_WARNING_BYTES && (
                  <span className={styles.sizeWarning}>Large file — conversion may take a while</span>
                )}
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
              {(['mp4', 'webm', 'gif'] as OutputFormat[]).map(f => (
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

          <fieldset className={styles.fieldset}>
            <legend className={styles.legend}>Resolution</legend>
            <div className={styles.formatToggle}>
              {(['original', '1080', '720', '480'] as Resolution[]).map(r => (
                <button
                  key={r}
                  className={`${styles.formatBtn} ${resolution === r ? styles.active : ''}`}
                  onClick={() => setResolution(r)}
                >
                  {resolutionLabel(r)}
                </button>
              ))}
            </div>
          </fieldset>

          {format !== 'gif' && (
            <fieldset className={styles.fieldset}>
              <legend className={styles.legend}>Quality (CRF) — {crf}</legend>
              <input
                type="range"
                min={18}
                max={35}
                step={1}
                value={crf}
                onChange={e => setCrf(Number(e.target.value))}
                className={styles.slider}
              />
              <div className={styles.sliderLabels}>
                <span>Higher quality</span>
                <span>Smaller file</span>
              </div>
            </fieldset>
          )}

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


export default VideoConverterPage
