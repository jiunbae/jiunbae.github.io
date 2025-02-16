import type { HeadProps } from 'gatsby'
import { ProfileCard, FloatingButton, Seo } from '@/components'

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


export const Head = ({ location: { pathname }, data: { site } }: HeadProps<Queries.HomeQuery>) => {
  const seo = {
    title: site?.siteMetadata.title,
    description: site?.siteMetadata.description,
    heroImage: ''
  }

  return <Seo title={seo.title} description={seo.description} heroImage={seo.heroImage} pathname={pathname}></Seo>
}

export default AboutPage
