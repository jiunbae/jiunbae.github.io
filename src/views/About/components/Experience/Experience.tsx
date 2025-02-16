import { useStaticQuery, graphql } from 'gatsby'

import * as styles from './Experience.module.scss'
import type { Job } from '../types'

export const ExperienceSection = () => {
  const { json: { experience: { jobs } } } = useStaticQuery(graphql`
    query {
      json {
        experience {
          jobs {
            company
            companyUrl
            team
            teamUrl
            position
            period
            description
          }
        }
      }
    }
  `)
  
  return (
    <section className={styles.experience}>
      <h2>Experience</h2>
      {jobs.map((job: Job, index: number) => (
        <div key={index} className={styles.job}>
          <h3>
            <span className={styles.title}>
              <a href={job.companyUrl} target="_blank" rel="noopener noreferrer">
                <strong>{job.company}</strong>
              </a>
              {job.team && (
                <span className={styles.team}>
                  <a href={job.teamUrl} target="_blank" rel="noopener noreferrer">
                    {job.team}
                  </a>
                </span>
              )}
            </span>
            <span className={styles.period}>{job.period}</span>
          </h3>
          <p className={styles.position}>
            <strong>
              {job.position}
            </strong>
          </p>
          {job.description && (
            <ul>
              {job.description.map((desc, i) => (
                <li key={i}>{desc}</li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </section>
  )
}
