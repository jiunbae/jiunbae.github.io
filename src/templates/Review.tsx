import { graphql } from 'gatsby'

export const query = graphql`
  query Review($id: String) {
    markdownRemark(id: { eq: $id }) {
      html
      excerpt(pruneLength: 200, truncate: true)
      frontmatter {
        title
        slug
        date(formatString: "YYYY년 MM월 DD일")
        dateISO: date(formatString: "YYYY-MM-DD")
        mediaType
        rating
        oneLiner
        tags
        poster {
          childImageSharp {
            gatsbyImageData(width: 300, placeholder: BLURRED)
          }
        }
        metadata {
          originalTitle
          year
          director
          creator
          author
          genre
          runtime
          pages
        }
      }
    }
  }
`

export { Head } from '../views/Review'
export { default } from '../views/Review'
