import { graphql } from 'gatsby'

export const query = graphql`
  query NoteTemplate($id: String!) {
    markdownRemark(id: { eq: $id }) {
      id
      html
      excerpt(pruneLength: 200, truncate: true)
      frontmatter {
        title
        date(formatString: "YY.MM.DD")
        tags
        slug
        description
      }
    }
  }
`

export { Head } from '../views/Note'
export { default } from '../views/Note'
