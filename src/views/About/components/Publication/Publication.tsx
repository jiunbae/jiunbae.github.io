import { useStaticQuery, graphql } from 'gatsby'

import * as styles from './Publication.module.scss'
import type { Publication } from '../types'

export const PublicationSection = () => {
  const { site: { siteMetadata: { name } }, json: { publications } } = useStaticQuery(graphql`
    query {
      site {
        siteMetadata {
          name {
            en
          }
        }
      }
      json {
        publications {
          papers {
            title
            authors
            journal
            volume
            year
            url
          }
        }
      }
    }
  `)

  return (
    <section className={styles.publication}>
      <h2>Publications</h2>
      {publications.papers.map((paper: Publication, index: number) => (
        <div key={index} className={styles.paper}>
          <h3 className={styles.title}>
            {paper.url ? (
              <a href={paper.url} target="_blank" rel="noopener noreferrer">
                <strong>{paper.title}</strong>
              </a>
            ) : (
              <strong>{paper.title}</strong>
            )}
          </h3>
          <p className={styles.authors}>
            {paper.authors.map((author, i) => (
              <span key={i}>
                {author === name.en ? <strong>{author}</strong> : author}
                {i < paper.authors.length - 1 ? ', ' : ''}
              </span>
            ))}
          </p>
          <p className={styles.meta}>
            {paper.journal}
            {paper.volume && `, ${paper.volume}`}, {paper.year}
          </p>
        </div>
      ))}
    </section>
  )
} 