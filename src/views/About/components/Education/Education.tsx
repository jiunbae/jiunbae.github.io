import { useStaticQuery, graphql } from 'gatsby'

import * as styles from './Education.module.scss'
import type { School } from '../types'

export const EducationSection = () => {
  const { json: { education } } = useStaticQuery(graphql`
    query {
      json {
        education {
          schools {
            name
            location
            degree
            period
            url
            status
            description
            advisor {
              name
              url
            }
          }
        }
      }
    }
  `)

  return (
    <section className={styles.education}>
      <h2>Education</h2>
      {education.schools.map((school: School, index: number) => (
        <div key={index} className={styles.school}>
          <h3>
            <span className={styles.title}>
              <a href={school.url} target="_blank" rel="noopener noreferrer">
                <strong>{school.name}</strong>
              </a>
            </span>
            <div className={styles.meta}>
              <span className={styles.location}>{school.location}</span>
              <span className={styles.period}>{school.period}</span>
            </div>
          </h3>
          <p className={styles.degree}>
            <strong>{school.degree}</strong>
            {school.advisor && (
              <span className={styles.advisor}>
                <a href={school.advisor.url} target="_blank" rel="noopener noreferrer">
                  {school.advisor.name}
                </a>
              </span>
            )}
            {school.status && (
              <span className={styles.status}>{school.status}</span>
            )}
          </p>
          {school.description && (
            <p className={styles.description}>{school.description}</p>
          )}
        </div>
      ))}
    </section>
  )
}
