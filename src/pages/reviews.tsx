import { graphql } from 'gatsby'

export const query = graphql`
  query Reviews {
    allMarkdownRemark(
      sort: { frontmatter: { date: DESC } }
      filter: {
        fields: { collection: { eq: "review" } }
      }
    ) {
      totalCount
      nodes {
        id
        frontmatter {
          title
          slug
          date(formatString: "YY.MM.DD")
          sortDate: date
          mediaType
          rating
          oneLiner
          tags
          published
          poster {
            childImageSharp {
              gatsbyImageData(width: 500, placeholder: BLURRED)
            }
          }
          metadata {
            year
            genre
          }
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

export { Head } from '../views/Reviews'
export { default } from '../views/Reviews'
