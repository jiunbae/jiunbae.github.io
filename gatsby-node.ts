import type { CreateSchemaCustomizationArgs, CreateWebpackConfigArgs, GatsbyNode } from 'gatsby'
import path from 'path'
import { promises as fsPromises, Dirent } from 'fs'
import { Resvg, type ResvgRenderOptions } from '@resvg/resvg-js'
import { decompress } from 'wawoff2'
import { createRequire } from 'module'

const requireFromNode = createRequire(__filename)

const sanitizeNoteSlug = (slug: string) => {
  const trimmed = slug.replace(/^\/+/, '').replace(/\/+$/, '')
  return trimmed.replace(/[^a-zA-Z0-9-_]/g, '-').replace(/-+/g, '-').toLowerCase() || 'note'
}

const OG_IMAGE_WIDTH = 1200
const OG_IMAGE_HEIGHT = 630

type OgFontConfig = {
  family: string;
  stack: string[];
  titleWeight: number;
  bodyWeight: number;
  fontFiles: string[];
  fontBuffers: Buffer[];
};

const loadOgFonts = (() => {
  let cache: OgFontConfig | null = null
  return async () => {
    if (cache) return cache

    const searchDirs = [
      '/usr/share/fonts/opentype',
      '/usr/share/fonts/truetype',
      '/usr/share/fonts',
      '/System/Library/Fonts',
      '/Library/Fonts',
      path.join(process.env.HOME ?? '', 'Library/Fonts')
    ]

    const findFontFile = async (patterns: RegExp[], directories: string[], depth = 3): Promise<string | null> => {
      for (const directory of directories) {
        let entries: Dirent[]

        try {
          entries = await fsPromises.readdir(directory, { withFileTypes: true })
        } catch (error) {
          const err = error as NodeJS.ErrnoException
          if (err.code && err.code !== 'ENOENT' && err.code !== 'ENOTDIR') throw err
          continue
        }

        for (const entry of entries) {
          const entryPath = path.join(directory, entry.name)

          if (entry.isDirectory()) {
            if (depth > 0) {
              const nested = await findFontFile(patterns, [entryPath], depth - 1)
              if (nested) return nested
            }
            continue
          }

          if (patterns.some(pattern => pattern.test(entry.name))) {
            return entryPath
          }
        }
      }

      return null
    }

    const systemRegularPath = await findFontFile([
      /NotoSansCJK.*Regular\.(ttc|otf|ttf)$/i,
      /NotoSansKR-?Regular\.(ttc|otf|ttf)$/i,
      /AppleSDGothicNeo-Regular\.(ttc|otf|ttf)$/i,
      /MalgunGothic\.ttf$/i
    ], searchDirs)

    const systemBoldPath = await findFontFile([
      /NotoSansCJK.*Bold\.(ttc|otf|ttf)$/i,
      /NotoSansKR-?Bold\.(ttc|otf|ttf)$/i,
      /AppleSDGothicNeo-Bold\.(ttc|otf|ttf)$/i,
      /MalgunGothicBold\.ttf$/i
    ], searchDirs)

    if (systemRegularPath || systemBoldPath) {
      const uniquePaths = Array.from(
        new Set([systemRegularPath, systemBoldPath].filter((value): value is string => Boolean(value)))
      )
      console.info(`[notes][og] system font 사용: ${uniquePaths.join(', ')}`)
      cache = {
        family: 'Noto Sans CJK KR',
        stack: ['Noto Sans CJK KR', 'Noto Sans CJK KR Regular', 'Noto Sans KR', 'Apple SD Gothic Neo', 'sans-serif'],
        titleWeight: systemBoldPath ? 700 : 500,
        bodyWeight: 400,
        fontFiles: uniquePaths,
        fontBuffers: []
      }

      return cache
    }

    const bundledFontSources = [
      {
        weight: 700,
        path: path.join(process.cwd(), 'static', 'fonts', 'noto-sans-kr-korean-700-normal.ttf')
      },
      {
        weight: 400,
        path: path.join(process.cwd(), 'static', 'fonts', 'noto-sans-kr-korean-400-normal.ttf')
      }
    ]

    const bundledFiles: string[] = []
    let bundledHasBold = false
    let bundledHasRegular = false

    for (const source of bundledFontSources) {
      try {
        await fsPromises.access(source.path)
        bundledFiles.push(source.path)
        if (source.weight >= 600) {
          bundledHasBold = true
        } else {
          bundledHasRegular = true
        }
        console.info(`[notes][og] bundled font 사용 (weight: ${source.weight}) ${source.path}`)
      } catch (error) {
        const err = error as NodeJS.ErrnoException
        if (err.code !== 'ENOENT') {
          const reason = err.message ?? String(err)
          console.warn(`[notes][og] bundled 폰트 접근 실패 (${source.path})\n${reason}`)
        }
      }
    }

    if (bundledFiles.length > 0) {
      cache = {
        family: 'Noto Sans KR',
        stack: ['Noto Sans KR', 'sans-serif'],
        titleWeight: bundledHasBold ? 700 : 500,
        bodyWeight: bundledHasRegular ? 400 : 500,
        fontFiles: bundledFiles,
        fontBuffers: []
      }

      return cache
    }

    const fallbackFontSources = [
      {
        weight: 700,
        modulePath: '@fontsource/noto-sans-kr/files/noto-sans-kr-korean-700-normal.woff2',
        fallback: '@fontsource/noto-sans-kr/files/noto-sans-kr-korean-700-normal.woff'
      },
      {
        weight: 400,
        modulePath: '@fontsource/noto-sans-kr/files/noto-sans-kr-korean-400-normal.woff2',
        fallback: '@fontsource/noto-sans-kr/files/noto-sans-kr-korean-400-normal.woff'
      }
    ]

    const buffers: Buffer[] = []
    let hasBold = false
    let hasRegular = false

    for (const source of fallbackFontSources) {
      const resolved = (() => {
        try {
          return requireFromNode.resolve(source.modulePath)
        } catch (error) {
          try {
            return requireFromNode.resolve(source.fallback)
          } catch (fallbackError) {
            return null
          }
        }
      })()

      if (!resolved) {
        console.warn(`[notes][og] @fontsource Noto Sans KR 폰트 모듈을 찾지 못했습니다: ${source.modulePath}`)
        continue
      }

      if (!resolved.endsWith('.woff2')) {
        console.warn(`[notes][og] @fontsource Noto Sans KR woff2 자산이 필요합니다. 사용 가능한 파일: ${resolved}`)
        continue
      }

      try {
        const fontData = await fsPromises.readFile(resolved)
        const decompressed = await decompress(fontData)
        buffers.push(Buffer.from(decompressed))
        if (source.weight >= 600) {
          hasBold = true
        } else {
          hasRegular = true
        }
        console.info(`[notes][og] @fontsource Noto Sans KR 폴백 사용 (weight: ${source.weight})`)
      } catch (error) {
        const reason = error instanceof Error ? error.message : String(error)
        console.warn(`[notes][og] @fontsource Noto Sans KR woff2 디코딩 실패 (${resolved})\n${reason}`)
      }
    }

    if (buffers.length > 0) {
      cache = {
        family: 'Noto Sans KR',
        stack: ['Noto Sans KR', 'Pretendard', 'sans-serif'],
        titleWeight: hasBold ? 600 : 500,
        bodyWeight: hasRegular ? 400 : 500,
        fontFiles: [],
        fontBuffers: buffers
      }

      return cache
    }

    throw new Error('[notes][og] 사용할 한글 글꼴을 찾지 못했습니다. 시스템 글꼴 또는 @fontsource woff2 자산을 확인하세요.')
  }
})()

const wrapText = (text: string, maxCharsPerLine: number, maxLines: number) => {
  const words = text.replace(/\s+/g, ' ').trim().split(' ')
  const lines: string[] = []
  let currentLine = ''

  for (const word of words) {
    const nextLine = currentLine.length === 0 ? word : `${currentLine} ${word}`
    if (nextLine.length > maxCharsPerLine && currentLine.length > 0) {
      lines.push(currentLine)
      currentLine = word
    } else {
      currentLine = nextLine
    }

    if (lines.length === maxLines) break
  }

  if (lines.length < maxLines && currentLine) {
    lines.push(currentLine)
  }

  if (lines.length > maxLines) {
    lines.length = maxLines
    lines[maxLines - 1] = `${lines[maxLines - 1].slice(0, maxCharsPerLine - 1).trim()}…`
  }

  return lines
}

const truncateSummary = (text: string, maxChars = 220, maxLines = 3) => {
  const normalized = text.replace(/\s+/g, ' ').trim()
  const truncated = normalized.length > maxChars ? `${normalized.slice(0, maxChars - 1).trim()}…` : normalized
  return wrapText(truncated, 34, maxLines)
}

const escapeXml = (value: string) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

const escapeFontFamilyName = (family: string) =>
  family.replace(/\\/g, '\\\\').replace(/'/g, "\\'")

const formatFontStack = (families: string[]) =>
  families
    .map(family => {
      if (family === 'sans-serif' || family === 'serif' || family === 'monospace') return family
      const escaped = escapeFontFamilyName(family)
      return /[\s,]/.test(escaped) ? `'${escaped}'` : escaped
    })
    .join(', ')

const createNoteOgSvg = (title: string, summary: string, date: string, fonts: OgFontConfig) => {
  const titleLines = wrapText(title, 16, 2)
  const summaryLines = truncateSummary(summary)

  const fontStack = formatFontStack(fonts.stack)

  const titleSpans = titleLines
    .map((line, index) => `<tspan x="96" dy="${index === 0 ? 0 : 68}">${escapeXml(line)}</tspan>`)
    .join('')

  const summarySpans = summaryLines
    .map((line, index) => `<tspan x="96" dy="${index === 0 ? 0 : 46}">${escapeXml(line)}</tspan>`)
    .join('')

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${OG_IMAGE_WIDTH}" height="${OG_IMAGE_HEIGHT}" viewBox="0 0 ${OG_IMAGE_WIDTH} ${OG_IMAGE_HEIGHT}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="og-bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1e293b" />
    </linearGradient>
  </defs>
  <rect fill="url(#og-bg)" width="${OG_IMAGE_WIDTH}" height="${OG_IMAGE_HEIGHT}" rx="32" />
  <text x="96" y="168" fill="#f8fafc" font-family="${fontStack}" font-size="58" font-weight="${fonts.titleWeight}">
    ${titleSpans}
  </text>
  <text x="96" y="328" fill="rgba(248, 250, 252, 0.9)" font-family="${fontStack}" font-size="30" font-weight="${fonts.bodyWeight}">
    ${summarySpans}
  </text>
  <g font-family="${fontStack}" font-size="26" font-weight="${fonts.bodyWeight}" fill="rgba(248, 250, 252, 0.68)">
    <text x="96" y="536">notes.jiun.dev</text>
    <text x="${OG_IMAGE_WIDTH - 96}" y="536" text-anchor="end">${escapeXml(date)}</text>
  </g>
</svg>`
}

const generateNoteOgImage = async ({
  title,
  date,
  slug,
  description,
  excerpt
}: {
  title: string;
  date: string;
  slug: string;
  description?: string | null;
  excerpt?: string | null;
}) => {
  const normalizedSlug = sanitizeNoteSlug(slug)
  const outputDir = path.join(process.cwd(), 'public', 'og', 'notes')
  await fsPromises.mkdir(outputDir, { recursive: true })

  const summaryText = description?.trim() || excerpt?.trim() || '짧은 생각을 기록하는 Notes'

  const fonts = await loadOgFonts()
  const svg = createNoteOgSvg(title, summaryText, date, fonts)
  const fontOptions: Record<string, unknown> = {
    loadSystemFonts: false,
    defaultFontFamily: fonts.family,
    sansSerifFamily: fonts.family
  }

  if (fonts.fontBuffers.length > 0) {
    fontOptions.fontBuffers = fonts.fontBuffers
  } else if (fonts.fontFiles.length > 0) {
    fontOptions.fontFiles = fonts.fontFiles
  }

  const resvg = new Resvg(svg, {
    fitTo: { mode: 'width', value: OG_IMAGE_WIDTH },
    font: fontOptions as ResvgRenderOptions['font'],
    languages: ['en', 'ko']
  })

  const pngData = resvg.render().asPng()
  const outputPath = path.join(outputDir, `${normalizedSlug}.png`)
  await fsPromises.writeFile(outputPath, pngData)
}

export const createPages: GatsbyNode['createPages'] = async ({ graphql, actions }) => {
  const { createPage } = actions
  const blogPostTemplate = path.resolve('src/templates/Post.tsx')
  const noteTemplate = path.resolve('src/templates/Note.tsx')
  const reviewTemplate = path.resolve('src/templates/Review.tsx')

  const result = await graphql<Queries.PagesQuery>(`
    query Pages {
      allMarkdownRemark(
        sort: { frontmatter: { date: DESC } }
        filter: { fields: { collection: { eq: "post" } } }
      ) {
        edges {
          node {
            frontmatter {
              slug
            }
            id
          }
          previous {
            frontmatter {
              slug
              title
            }
          }
          next {
            frontmatter {
              slug
              title
            }
          }
        }
      }
      notes: allMarkdownRemark(
        sort: { frontmatter: { date: DESC } }
        filter: { fields: { collection: { eq: "note" } } }
      ) {
        nodes {
          id
          frontmatter {
            slug
            title
          }
        }
      }
      reviews: allMarkdownRemark(
        sort: { frontmatter: { date: DESC } }
        filter: { fields: { collection: { eq: "review" } } }
      ) {
        nodes {
          id
          frontmatter {
            slug
            title
          }
        }
      }
    }
  `)

  if (result.errors) {
    throw result.errors
  }

  const posts = result.data?.allMarkdownRemark.edges
  const notes = result.data?.notes?.nodes ?? []
  const reviews = result.data?.reviews?.nodes ?? []

  posts?.forEach(({ node, previous, next }) => {
    createPage({
      path: `/posts${node.frontmatter.slug}`,
      component: blogPostTemplate,
      context: {
        id: node.id,
        frontmatter__slug: node.frontmatter.slug,
        previous: previous === null ? null : previous.frontmatter.slug,
        previousTitle: previous === null ? null : previous.frontmatter.title,
        next: next === null ? null : next.frontmatter.slug,
        nextTitle: next === null ? null : next.frontmatter.title
      }
    })
  })

  notes.forEach(note => {
    const slug = note.frontmatter.slug

    if (!slug) {
      throw new Error(`노트 슬러그가 존재하지 않습니다. (id: ${note.id})`)
    }

    const sanitizedSlug = sanitizeNoteSlug(slug)
    const notePath = `/notes/${sanitizedSlug}/`

    createPage({
      path: notePath,
      component: noteTemplate,
      context: {
        id: note.id
      }
    })
  })

  reviews.forEach(review => {
    const slug = review.frontmatter.slug

    if (!slug) {
      throw new Error(`리뷰 슬러그가 존재하지 않습니다. (id: ${review.id})`)
    }

    createPage({
      path: slug,
      component: reviewTemplate,
      context: {
        id: review.id
      }
    })
  })
}

export const onCreateWebpackConfig: GatsbyNode['onCreateWebpackConfig'] = ({ actions }: CreateWebpackConfigArgs) => {
  actions.setWebpackConfig({
    resolve: {
      alias: {
        '@/components': path.resolve(__dirname, 'src/components'),
        '@/images': path.resolve(__dirname, 'src/images'),
        '@/styles': path.resolve(__dirname, 'src/styles'),
        '@/utils': path.resolve(__dirname, 'src/utils'),
        '@/contexts': path.resolve(__dirname, 'src/contexts'),
        '@/layouts': path.resolve(__dirname, 'src/layouts')
      }
    }
  })
}

export const onCreateNode: GatsbyNode['onCreateNode'] = ({ node, actions, getNode }) => {
  if (node.internal.type !== 'MarkdownRemark') return

  if (!node.parent) return

  const parent = getNode(node.parent)

  if (!parent || parent.internal.type !== 'File') return

  const { sourceInstanceName } = parent as typeof parent & {
    sourceInstanceName?: string | null;
  }

  if (sourceInstanceName !== 'notes' && sourceInstanceName !== 'posts' && sourceInstanceName !== 'reviews') return

  const collection =
    sourceInstanceName === 'notes' ? 'note' :
    sourceInstanceName === 'reviews' ? 'review' :
    'post'

  actions.createNodeField({
    node,
    name: 'collection',
    value: collection
  })

  // published 필드가 없으면 기본값 true 설정
  const frontmatter = (node as any).frontmatter
  if (frontmatter && frontmatter.published === undefined) {
    frontmatter.published = true
  }
}

export const createSchemaCustomization: GatsbyNode['createSchemaCustomization'] = ({
  actions
}: CreateSchemaCustomizationArgs) => {
  const { createTypes } = actions

  createTypes(`
    type SiteSiteMetadata {
      title: String!
      siteUrl: String!
      description: String!
      heroImage: String!
      keywords: [String!]!
    }

    type Site implements Node {
      siteMetadata: SiteSiteMetadata!
    }

    type ReviewMetadata {
      originalTitle: String
      year: Int
      director: String
      creator: String
      author: String
      genre: [String!]
      runtime: String
      pages: String
    }

    type Frontmatter {
      title: String!
      description: String
      slug: String!
      date: Date! @dateformat
      tags: [String!]!
      heroImage: File @fileByRelativePath
      heroImageAlt: String
      poster: File @fileByRelativePath
      mediaType: String
      rating: Float
      oneLiner: String
      metadata: ReviewMetadata
      published: Boolean @dontInfer
    }

    type MarkdownRemarkFields {
      collection: String!
    }

    type MarkdownRemark implements Node {
      frontmatter: Frontmatter!
      id: String!
      html: String!
      fields: MarkdownRemarkFields!
    }
  `)
}

export const onPostBuild: GatsbyNode['onPostBuild'] = async ({ graphql, reporter }) => {
  const result = await graphql<{
    allMarkdownRemark: {
      nodes: {
        frontmatter: {
          title: string;
          slug: string;
          date: string;
          description?: string | null;
        };
        excerpt: string | null;
      }[];
    };
  }>(`
    query NoteOgImageSources {
      allMarkdownRemark(
        filter: { fields: { collection: { eq: "note" } } }
        sort: { frontmatter: { date: DESC } }
      ) {
        nodes {
          frontmatter {
            title
            slug
            date(formatString: "YYYY.MM.DD")
            description
          }
          excerpt(pruneLength: 220)
        }
      }
    }
  `)

  if (result.errors) {
    reporter.warn('노트 OG 이미지 생성을 위한 GraphQL 쿼리에 실패했습니다.')
    return
  }

  const notes = result.data?.allMarkdownRemark.nodes ?? []

  if (notes.length === 0) return

  await Promise.all(
    notes
      .filter(note => note.frontmatter.slug)
      .map(async note => {
        try {
          await generateNoteOgImage({
            title: note.frontmatter.title,
            date: note.frontmatter.date,
            slug: note.frontmatter.slug,
            description: note.frontmatter.description,
            excerpt: note.excerpt
          })
        } catch (error) {
          const reason = error instanceof Error ? (error.stack ?? error.message) : String(error)
          reporter.warn(`노트 OG 이미지 생성 중 오류가 발생했습니다. slug: ${note.frontmatter.slug}\n${reason}`)
        }
      })
  )

  reporter.info(`[notes] OG 이미지가 생성되었습니다. (${notes.length}건)`)
}
