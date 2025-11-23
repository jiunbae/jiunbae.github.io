import { useStaticQuery, graphql } from 'gatsby'

export interface SearchItem {
  slug: string
  title: string
  excerpt: string
  date: string
  type: 'post' | 'note' | 'review'
  tags?: readonly string[]
}

export const useSearchIndex = (): SearchItem[] => {
  const data = useStaticQuery<Queries.SearchIndexQuery>(graphql`
    query SearchIndex {
      allMarkdownRemark {
        nodes {
          fields {
            collection
          }
          frontmatter {
            title
            date(formatString: "YYYY-MM-DD")
            tags
            slug
          }
          excerpt(pruneLength: 200)
        }
      }
    }
  `)

  return data.allMarkdownRemark.nodes
    .filter(node => node.frontmatter?.slug && node.frontmatter?.title)
    .map(node => ({
      slug: node.frontmatter!.slug!,
      title: node.frontmatter!.title!,
      excerpt: node.excerpt || '',
      date: node.frontmatter!.date || '',
      type: (node.fields?.collection as 'post' | 'note' | 'review') || 'post',
      tags: node.frontmatter!.tags || [],
    }))
}
