import { Description, Heading } from './components'
import * as styles from './ProfileCard.module.scss'
import { useStaticQuery, graphql } from 'gatsby'

import { SocialIconList } from './components/SocialIconList/SocialIconList'

export const ProfileCard = () => {
  const { site: { siteMetadata : { name, social } } } = useStaticQuery(graphql`
    query {
      site {
        siteMetadata {
          name {
            kr
            en
          }
          social {
            email
            facebook
            linkedin
            github
            twitter
            instagram
          }
        }
      }
    }
  `)

  return (
    <div className={styles.card}>
      <div className={styles.mainContent}>
        <img
          className={styles.profileImage}
          src="/profile.png"
          alt="Profile"
        />
        <div className={styles.content}>
          <Heading text={name.en} subText={name.kr} />
          <div className={styles.info}>
            <SocialIconList social={social} />
          </div>
        </div>
      </div>
      <div className={styles.description}>
        <Description />
      </div>
    </div>
  )
}
