import { useStaticQuery, graphql } from 'gatsby'

import * as styles from './Awards.module.scss'
import type { Award } from '../types'

export const AwardsSection = () => {
  const { json: { awards } } = useStaticQuery(graphql`
    query {
      json {
        awards {
          awards {
            title
            year
          }
        }
      }
    }
  `)

  return (
    <section className={styles.awards}>
      <h2>Awards</h2>
      <ul>
        {awards.awards.map((award: Award, index: number) => (
          <li key={index}>
            <span className={styles.title}>{award.title}</span>
            <span className={styles.year}>{award.year}</span>
          </li>
        ))}
      </ul>
    </section>
  )
} 