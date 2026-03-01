import type { HeadProps } from 'gatsby'
import { Link } from 'gatsby'
import { Seo } from '@/components'
import toolItems from '@/data/tools'

import * as styles from './Tools.module.scss'

const ToolsPage = () => {
  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h1 className={styles.title}>Tools</h1>
        <p className={styles.subtitle}>
          Handy browser-based utilities — no uploads, everything runs locally
        </p>
      </div>

      {toolItems.length === 0 ? (
        <div className={styles.empty}>Coming soon...</div>
      ) : (
        <div className={styles.grid}>
          {toolItems.map(item => (
            <Link key={item.slug} to={item.slug} className={styles.card}>
              <div className={styles.iconArea}>
                <span className={styles.icon}>{item.icon}</span>
              </div>
              <div className={styles.cardBody}>
                <h2 className={styles.cardTitle}>{item.title}</h2>
                <p className={styles.cardDescription}>{item.description}</p>
                <div className={styles.tags}>
                  {item.tags.map(tag => (
                    <span key={tag} className={styles.tag}>{tag}</span>
                  ))}
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => (
  <Seo
    title="Tools"
    description="Browser-based utilities by Jiun Bae — image, audio, and video converters"
    heroImage=""
    pathname={pathname}
  />
)

export default ToolsPage
