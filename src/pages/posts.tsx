import { graphql } from 'gatsby'

export const query = graphql`
  query Posts {
    allMarkdownRemark(
      sort: { frontmatter: { date: DESC } }
      filter: {
        fields: { collection: { eq: "post" } }
      }
    ) {
      totalCount
      nodes {
        frontmatter {
          tags
          slug
          description
          date(formatString: "YY.MM.DD")
          title
          published
          heroImage {
            childImageSharp {
              gatsbyImageData(placeholder: BLURRED)
            }
          }
          heroImageAlt
        }
        excerpt(pruneLength: 160)
        id
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

export { Head } from '../views/Home'
export { default } from '../views/Home'
