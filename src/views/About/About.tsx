import { ProfileCard, FloatingButton } from '@/components'

import { EducationSection, ExperienceSection, ProjectsSection, AwardsSection, SkillsSection, PublicationSection } from './components'
import * as styles from './About.module.scss'

const AboutPage = () => {
  return (
    <main className={styles.page}>
      <ProfileCard />
      <div className={styles.content}>
        <ExperienceSection />
        <ProjectsSection />
        <EducationSection />
        <AwardsSection />
        <PublicationSection />
        <SkillsSection />
      </div>
      <FloatingButton />
    </main>
  )
}

export default AboutPage

export const Head = () => <title>About</title>
