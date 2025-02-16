import { useStaticQuery, graphql } from 'gatsby'

import * as styles from './Projects.module.scss'
import type { Project } from '../types'

export const ProjectsSection = () => {
  const { json: { projects } } = useStaticQuery(graphql`
    query {
      json {
        projects {
          projects {
            title
            url
            organization
            period
            description
          }
        }
      }
    }
  `)

  return (
    <section className={styles.projects}>
      <h2>Projects</h2>
      {projects.projects.map((project: Project, index: number) => (
        <div key={index} className={styles.project}>
          <h3>
            <span className={styles.title}>
              <a href={project.url} target="_blank" rel="noopener noreferrer">
                <strong>{project.title}</strong>
              </a>
            </span>
            <div className={styles.meta}>
              <span className={styles.organization}>{project.organization}</span>
              <span className={styles.period}>{project.period}</span>
            </div>
          </h3>
          <ul>
            {project.description.map((desc, i) => (
              <li key={i}>{desc}</li>
            ))}
          </ul>
        </div>
      ))}
    </section>
  )
} 