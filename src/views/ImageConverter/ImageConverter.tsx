import { useState, useRef, useCallback, useEffect } from 'react'
import type { HeadProps } from 'gatsby'
import { Link } from 'gatsby'
import { Seo } from '@/components'

import * as styles from './ImageConverter.module.scss'

type Format = 'png' | 'jpeg' | 'webp' | 'pdf'
type Scale = 1 | 2 | 3 | 4

interface ImageDimensions {
  width: number
  height: number
}

const ACCEPT_TYPES = '.svg,.png,.jpg,.jpeg,.webp,.bmp,.gif,.tiff,.tif,image/*'

const parseSvgDimensions = (svgText: string): ImageDimensions | null => {
  const parser = new DOMParser()
  const doc = parser.parseFromString(svgText, 'image/svg+xml')
  const svg = doc.querySelector('svg')
  if (!svg) return null

  const viewBox = svg.getAttribute('viewBox')
  const w = svg.getAttribute('width')
  const h = svg.getAttribute('height')

  if (w && h) {
    const pw = parseFloat(w)
    const ph = parseFloat(h)
    if (pw > 0 && ph > 0) return { width: pw, height: ph }
  }

  if (viewBox) {
    const parts = viewBox.split(/[\s,]+/).map(Number)
    if (parts.length === 4 && parts[2] > 0 && parts[3] > 0) {
      return { width: parts[2], height: parts[3] }
    }
  }

  return { width: 300, height: 150 }
}

const isSvgFile = (file: File) =>
  file.type === 'image/svg+xml' || file.name.endsWith('.svg')

const formatLabel = (fmt: Format) => {
  switch (fmt) {
    case 'png': return 'PNG'
    case 'jpeg': return 'JPG'
    case 'webp': return 'WebP'
    case 'pdf': return 'PDF'
  }
}

const formatExt = (fmt: Format) => {
  switch (fmt) {
    case 'jpeg': return 'jpg'
    case 'pdf': return 'pdf'
    default: return fmt
  }
}

const ImageConverterPage = () => {
  const [isSSR, setIsSSR] = useState(true)
  const [svgText, setSvgText] = useState<string | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [fileName, setFileName] = useState('')
  const [fileExt, setFileExt] = useState('')
  const [naturalSize, setNaturalSize] = useState<ImageDimensions>({ width: 300, height: 150 })
  const [width, setWidth] = useState(300)
  const [height, setHeight] = useState(150)
  const [lockAspect, setLockAspect] = useState(true)
  const [scale, setScale] = useState<Scale>(1)
  const [format, setFormat] = useState<Format>('png')
  const [bgColor, setBgColor] = useState('#ffffff')
  const [transparent, setTransparent] = useState(true)
  const [quality, setQuality] = useState(0.92)
  const [isDragging, setIsDragging] = useState(false)
  const [isConverting, setIsConverting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isSvg, setIsSvg] = useState(false)

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

  const loadImage = useCallback((file: File) => {
    const baseName = file.name.replace(/\.[^.]+$/, '')
    const ext = file.name.split('.').pop() || ''
    setFileName(baseName)
    setFileExt(ext)

    if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current)

    if (isSvgFile(file)) {
      setIsSvg(true)
      const reader = new FileReader()
      reader.onload = () => {
        const text = reader.result as string
        setSvgText(text)

        const dims = parseSvgDimensions(text)
        if (dims) {
          setNaturalSize(dims)
          setWidth(Math.round(dims.width))
          setHeight(Math.round(dims.height))
        }

        const blob = new Blob([text], { type: 'image/svg+xml;charset=utf-8' })
        const url = URL.createObjectURL(blob)
        prevUrlRef.current = url
        setPreviewUrl(url)
      }
      reader.readAsText(file)
    } else {
      setIsSvg(false)
      setSvgText(null)
      const url = URL.createObjectURL(file)
      prevUrlRef.current = url

      const img = new Image()
      img.onload = () => {
        const dims = { width: img.naturalWidth, height: img.naturalHeight }
        setNaturalSize(dims)
        setWidth(dims.width)
        setHeight(dims.height)
        setPreviewUrl(url)
      }
      img.onerror = () => {
        URL.revokeObjectURL(url)
        prevUrlRef.current = null
      }
      img.src = url
    }
  }, [])

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) loadImage(file)
  }, [loadImage])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) loadImage(file)
  }, [loadImage])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleWidthChange = useCallback((val: number) => {
    setWidth(val)
    if (lockAspect && naturalSize.width > 0) {
      setHeight(Math.round(val * (naturalSize.height / naturalSize.width)))
    }
  }, [lockAspect, naturalSize])

  const handleHeightChange = useCallback((val: number) => {
    setHeight(val)
    if (lockAspect && naturalSize.height > 0) {
      setWidth(Math.round(val * (naturalSize.width / naturalSize.height)))
    }
  }, [lockAspect, naturalSize])

  const handleRemove = useCallback(() => {
    setSvgText(null)
    setPreviewUrl(null)
    setFileName('')
    setFileExt('')
    setIsSvg(false)
    setError(null)
    if (prevUrlRef.current) {
      URL.revokeObjectURL(prevUrlRef.current)
      prevUrlRef.current = null
    }
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [])

  const handleConvert = useCallback(async () => {
    if (!previewUrl) return
    setIsConverting(true)
    setError(null)

    try {
      const outputW = width * scale
      const outputH = height * scale

      // Load source image
      let imgSrc = previewUrl
      if (isSvg && svgText) {
        const blob = new Blob([svgText], { type: 'image/svg+xml;charset=utf-8' })
        imgSrc = URL.createObjectURL(blob)
      }

      const img = new Image()
      img.crossOrigin = 'anonymous'

      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve()
        img.onerror = () => reject(new Error('Failed to load image'))
        img.src = imgSrc
      }).finally(() => {
        if (isSvg && imgSrc !== previewUrl) URL.revokeObjectURL(imgSrc)
      })

      const canvas = document.createElement('canvas')
      canvas.width = outputW
      canvas.height = outputH
      const ctx = canvas.getContext('2d')
      if (!ctx) throw new Error('Failed to create canvas context')

      // Fill background for formats without transparency
      if (format === 'jpeg' || format === 'pdf' || !transparent) {
        ctx.fillStyle = bgColor
        ctx.fillRect(0, 0, outputW, outputH)
      }

      ctx.drawImage(img, 0, 0, outputW, outputH)

      if (format === 'pdf') {
        const { default: jsPDF } = await import('jspdf')
        const orientation = outputW >= outputH ? 'landscape' : 'portrait'
        const pdf = new jsPDF({ orientation, unit: 'px', format: [outputW, outputH] })
        const dataUrl = canvas.toDataURL('image/png')
        pdf.addImage(dataUrl, 'PNG', 0, 0, outputW, outputH)
        pdf.save(`${fileName || 'converted'}.pdf`)
        setIsConverting(false)
        return
      }

      const mimeType = format === 'jpeg' ? 'image/jpeg' : format === 'webp' ? 'image/webp' : 'image/png'
      const q = format === 'png' ? undefined : quality

      canvas.toBlob(
        (resultBlob) => {
          if (!resultBlob) {
            setError('Failed to generate image — try a smaller size')
            setIsConverting(false)
            return
          }
          const downloadUrl = URL.createObjectURL(resultBlob)
          const a = document.createElement('a')
          a.href = downloadUrl
          a.download = `${fileName || 'converted'}.${formatExt(format)}`
          a.click()
          URL.revokeObjectURL(downloadUrl)
          setIsConverting(false)
        },
        mimeType,
        q
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Conversion failed')
      setIsConverting(false)
    }
  }, [previewUrl, svgText, isSvg, width, height, scale, format, bgColor, transparent, quality, fileName])

  const showQuality = format === 'jpeg' || format === 'webp' || format === 'pdf'
  const showTransparency = format === 'png' || format === 'webp'

  if (isSSR) return null

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <Link to="/tools/" className={styles.backLink}>Tools</Link>
        <h1 className={styles.title}>Image Converter</h1>
        <p className={styles.subtitle}>
          Convert images between PNG, JPG, WebP, and PDF — runs entirely in your browser
        </p>
      </div>

      <div className={styles.layout}>
        {/* Left: Upload + Preview */}
        <div className={styles.previewSection}>
          {!previewUrl ? (
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
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <p className={styles.dropText}>Drop an image file here or click to browse</p>
              <p className={styles.dropHint}>SVG, PNG, JPG, WebP, BMP, GIF, TIFF</p>
              <input
                ref={fileInputRef}
                type="file"
                accept={ACCEPT_TYPES}
                onChange={handleFileChange}
                className={styles.hiddenInput}
              />
            </div>
          ) : (
            <div className={styles.previewArea}>
              <div className={styles.previewToolbar}>
                <span className={styles.fileName}>{fileName}.{fileExt}</span>
                <button className={styles.removeBtn} onClick={handleRemove}>
                  Remove
                </button>
              </div>
              <div
                className={styles.previewCanvas}
                style={{ backgroundImage: `url(${previewUrl})` }}
              />
            </div>
          )}
          {error && <div className={styles.error}>{error}</div>}
        </div>

        {/* Right: Controls */}
        <div className={styles.controls}>
          <fieldset className={styles.fieldset}>
            <legend className={styles.legend}>Format</legend>
            <div className={styles.formatToggle}>
              {(['png', 'jpeg', 'webp', 'pdf'] as Format[]).map(f => (
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
            <legend className={styles.legend}>Dimensions</legend>
            <div className={styles.dimensionRow}>
              <label className={styles.dimLabel}>
                <span>W</span>
                <input
                  type="number"
                  min={1}
                  value={width}
                  onChange={e => handleWidthChange(Number(e.target.value))}
                  className={styles.dimInput}
                />
              </label>
              <button
                className={`${styles.lockBtn} ${lockAspect ? styles.locked : ''}`}
                onClick={() => setLockAspect(!lockAspect)}
                title={lockAspect ? 'Unlock aspect ratio' : 'Lock aspect ratio'}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  {lockAspect ? (
                    <><rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0 1 10 0v4" /></>
                  ) : (
                    <><rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0 1 9.9-1" /></>
                  )}
                </svg>
              </button>
              <label className={styles.dimLabel}>
                <span>H</span>
                <input
                  type="number"
                  min={1}
                  value={height}
                  onChange={e => handleHeightChange(Number(e.target.value))}
                  className={styles.dimInput}
                />
              </label>
            </div>
          </fieldset>

          <fieldset className={styles.fieldset}>
            <legend className={styles.legend}>Scale</legend>
            <div className={styles.scaleRow}>
              {([1, 2, 3, 4] as Scale[]).map(s => (
                <button
                  key={s}
                  className={`${styles.scaleBtn} ${scale === s ? styles.active : ''}`}
                  onClick={() => setScale(s)}
                >
                  {s}x
                </button>
              ))}
            </div>
            <div className={styles.outputSize}>
              Output: {width * scale} &times; {height * scale}px
            </div>
          </fieldset>

          <fieldset className={styles.fieldset}>
            <legend className={styles.legend}>Background</legend>
            {showTransparency && (
              <label className={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={transparent}
                  onChange={e => setTransparent(e.target.checked)}
                />
                <span>Transparent</span>
              </label>
            )}
            {(format === 'jpeg' || format === 'pdf' || !transparent) && (
              <div className={styles.colorRow}>
                <input
                  type="color"
                  value={bgColor}
                  onChange={e => setBgColor(e.target.value)}
                  className={styles.colorInput}
                />
                <input
                  type="text"
                  value={bgColor}
                  onChange={e => setBgColor(e.target.value)}
                  className={styles.colorText}
                  maxLength={7}
                />
              </div>
            )}
          </fieldset>

          {showQuality && (
            <fieldset className={styles.fieldset}>
              <legend className={styles.legend}>Quality — {Math.round(quality * 100)}%</legend>
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.01}
                value={quality}
                onChange={e => setQuality(Number(e.target.value))}
                className={styles.slider}
              />
            </fieldset>
          )}

          <button
            className={styles.downloadBtn}
            onClick={handleConvert}
            disabled={!previewUrl || isConverting}
          >
            {isConverting ? 'Converting...' : `Download ${formatLabel(format)}`}
          </button>
        </div>
      </div>
    </div>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => (
  <Seo
    title="Image Converter"
    description="Convert images between PNG, JPG, WebP, and PDF — custom dimensions, scale, background, and quality. Runs entirely in your browser."
    heroImage=""
    pathname={pathname}
  />
)

export default ImageConverterPage
