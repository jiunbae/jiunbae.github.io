import * as styles from './SkipLink.module.scss'

export const SkipLink = () => {
  return (
    <a href="#main-content" className={styles.skipLink}>
      본문으로 바로가기
    </a>
  )
}
