import { graphql } from 'gatsby'

export const query = graphql`
  query Notes {
    allMarkdownRemark(
      sort: { frontmatter: { date: DESC } }
      filter: {
        fields: { collection: { eq: "note" } }
      }
    ) {
      totalCount
      nodes {
        id
        html
        frontmatter {
          title
          date(formatString: "YY.MM.DD")
          tags
          slug
          published
        }
      }
      group(field: { frontmatter: { tags: SELECT } }) {
        fieldValue
        totalCount
      }
    }
    site {
      siteMetadata {
        siteUrl
        description
        title
        keywords
      }
    }
    file(relativePath: { eq: "cover.png" }) {
      publicURL
    }
  }
`

export { Head } from '../views/Notes'
export { default } from '../views/Notes'
