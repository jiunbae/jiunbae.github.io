import type { CreateSchemaCustomizationArgs, CreateWebpackConfigArgs, GatsbyNode } from 'gatsby'
import path from 'path'

const sanitizeNoteSlug = (slug: string) => {
  const trimmed = slug.replace(/^\/+/, '').replace(/\/+$/, '')
  return trimmed.replace(/[^a-zA-Z0-9-_]/g, '-').replace(/-+/g, '-').toLowerCase() || 'note'
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

  const { sourceInstanceName, relativePath } = parent as typeof parent & {
    sourceInstanceName?: string | null;
    relativePath?: string | null;
  }

  if (sourceInstanceName === 'contents' && relativePath?.startsWith('notes/')) {
    actions.deleteNode(node)
    return
  }

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
