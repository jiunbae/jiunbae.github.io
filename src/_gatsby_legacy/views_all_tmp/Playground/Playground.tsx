import type { HeadProps } from 'gatsby'
import { Link } from 'gatsby'
import { Seo } from '@/components'
import playgroundItems from '@/data/playground'

import * as styles from './Playground.module.scss'

const PlaygroundPage = () => {
  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h1 className={styles.title}>Playground</h1>
        <p className={styles.subtitle}>
          Interactive experiments, generative art, and creative coding projects
        </p>
      </div>

      {playgroundItems.length === 0 ? (
        <div className={styles.empty}>Coming soon...</div>
      ) : (
        <div className={styles.grid}>
          {playgroundItems.map(item => (
            <Link key={item.slug} to={item.slug} className={styles.card}>
              <div
                className={styles.thumbnail}
                style={{ background: item.gradient }}
              />
              <div className={styles.cardBody}>
                <h2 className={styles.cardTitle}>{item.title}</h2>
                <p className={styles.cardDescription}>{item.description}</p>
                <div className={styles.cardMeta}>
                  <div className={styles.tags}>
                    {item.tags.map(tag => (
                      <span key={tag} className={styles.tag}>{tag}</span>
                    ))}
                  </div>
                  <span className={styles.date}>{item.date}</span>
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
    title="Playground"
    description="Interactive experiments, generative art, and creative coding projects by Jiun Bae"
    heroImage=""
    pathname={pathname}
  />
)

export default PlaygroundPage
