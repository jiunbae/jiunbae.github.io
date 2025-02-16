import { useStaticQuery, graphql } from 'gatsby'

import * as styles from './Skills.module.scss'

export const SkillsSection = () => {
  const { json: { skills } } = useStaticQuery(graphql`
    query {
      json {
        skills {
          languages
          frameworks
          tools
        }
      }
    }
  `)

  const skillSections = [
    { title: 'Languages', items: skills.languages },
    { title: 'Frameworks', items: skills.frameworks },
    { title: 'Tools', items: skills.tools }
  ]

  return (
    <section className={styles.skills}>
      <h2>Skills</h2>
      {skillSections.map((section, index) => (
        <div key={index} className={styles.skillSection}>
          <h3>{section.title}</h3>
          <ul>
            {section.items.map((item: string, i: number) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </div>
      ))}
    </section>
  )
}
