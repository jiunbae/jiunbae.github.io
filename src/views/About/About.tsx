import type { HeadProps } from 'gatsby'
import { FloatingButton, JsonLd, ProfileCard, Seo, createBreadcrumbSchema, createPersonSchema } from '@/components'

import { EducationSection, ExperienceSection, ProjectsSection, AwardsSection, SkillsSection, PublicationSection } from './components'
import * as styles from './About.module.scss'

const SITE_URL = 'https://jiun.dev'

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

  const pageUrl = pathname.startsWith('http') ? pathname : `${SITE_URL}${pathname}`
  const sameAs = [
    'https://www.facebook.com/MayTryArk',
    'https://linkedin.com/in/jiunbae',
    'https://github.com/jiunbae',
    'https://twitter.com/baejiun',
    'https://instagram.com/bae.jiun'
  ]

  const personSchema = createPersonSchema({
    name: '배지운',
    alternateName: 'Jiun Bae',
    url: `${SITE_URL}/about`,
    jobTitle: 'Software Engineer',
    sameAs
  })

  const breadcrumbSchema = createBreadcrumbSchema([
    { name: 'Home', url: SITE_URL },
    { name: 'About', url: pageUrl }
  ])

  return (
    <>
      <Seo title={seo.title} description={seo.description} heroImage={seo.heroImage} pathname={pathname} />
      <JsonLd data={[personSchema, breadcrumbSchema]} />
    </>
  )
}

export default AboutPage
