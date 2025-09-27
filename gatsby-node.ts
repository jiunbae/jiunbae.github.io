import type { CreateSchemaCustomizationArgs, CreateWebpackConfigArgs, GatsbyNode } from 'gatsby'
import path from 'path'
import { promises as fsPromises } from 'fs'
import { Resvg } from '@resvg/resvg-js'
import { createRequire } from 'module'

const requireFromNode = createRequire(__filename)

const sanitizeNoteSlug = (slug: string) => {
  const trimmed = slug.replace(/^\/+/, '').replace(/\/+$/, '')
  return trimmed.replace(/[^a-zA-Z0-9-_]/g, '-').replace(/-+/g, '-').toLowerCase() || 'note'
}

const OG_IMAGE_WIDTH = 1200
const OG_IMAGE_HEIGHT = 630

type OgFonts = {
  bold: string;
  regular: string;
};

const loadOgFonts = (() => {
  let cache: OgFonts | null = null

  return async () => {
    if (cache) return cache

    const [bold, regular] = await Promise.all([
      fsPromises.readFile(requireFromNode.resolve('@fontsource/noto-sans-kr/files/noto-sans-kr-korean-600-normal.woff2')),
      fsPromises.readFile(requireFromNode.resolve('@fontsource/noto-sans-kr/files/noto-sans-kr-korean-400-normal.woff2'))
    ])

    const toDataUrl = (buffer: Buffer) => `data:font/woff2;base64,${buffer.toString('base64')}`

    cache = {
      bold: toDataUrl(bold),
      regular: toDataUrl(regular)
    }

    return cache
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

const createNoteOgSvg = async (title: string, summary: string, date: string) => {
  const fonts = await loadOgFonts()
  const titleLines = wrapText(title, 16, 2)
  const summaryLines = truncateSummary(summary)

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
    <style>
      @font-face {
        font-family: 'NotesOG';
        font-style: normal;
        font-weight: 600;
        src: url(${fonts.bold}) format('woff2');
      }
      @font-face {
        font-family: 'NotesOG';
        font-style: normal;
        font-weight: 400;
        src: url(${fonts.regular}) format('woff2');
      }
    </style>
  </defs>
  <rect fill="url(#og-bg)" width="${OG_IMAGE_WIDTH}" height="${OG_IMAGE_HEIGHT}" rx="32" />
  <text x="96" y="168" fill="#f8fafc" font-family="NotesOG, 'Noto Sans KR', sans-serif" font-size="58" font-weight="600">
    ${titleSpans}
  </text>
  <text x="96" y="328" fill="rgba(248, 250, 252, 0.9)" font-family="NotesOG, 'Noto Sans KR', sans-serif" font-size="30" font-weight="400">
    ${summarySpans}
  </text>
  <g font-family="NotesOG, 'Noto Sans KR', sans-serif" font-size="26" fill="rgba(248, 250, 252, 0.68)">
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

  const svg = await createNoteOgSvg(title, summaryText, date)
  const resvg = new Resvg(svg, {
    fitTo: { mode: 'width', value: OG_IMAGE_WIDTH }
  })

  const pngData = resvg.render().asPng()
  const outputPath = path.join(outputDir, `${normalizedSlug}.png`)
  await fsPromises.writeFile(outputPath, pngData)
}

export const createPages: GatsbyNode['createPages'] = async ({ graphql, actions }) => {
  const { createPage } = actions
  const blogPostTemplate = path.resolve('src/templates/Post.tsx')
  const noteTemplate = path.resolve('src/templates/Note.tsx')

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
    }
  `)

  if (result.errors) {
    throw result.errors
  }

  const posts = result.data?.allMarkdownRemark.edges
  const notes = result.data?.notes?.nodes ?? []

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

  if (sourceInstanceName !== 'notes' && sourceInstanceName !== 'posts') return

  const collection = sourceInstanceName === 'notes' ? 'note' : 'post'

  actions.createNodeField({
    node,
    name: 'collection',
    value: collection
  })
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

    type Frontmatter {
      title: String!
      description: String
      slug: String!
      date: Date! @dateformat
      tags: [String!]!
      heroImage: File @fileByRelativePath
      heroImageAlt: String
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
