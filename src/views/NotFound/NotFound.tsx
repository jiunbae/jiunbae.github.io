import { Link } from 'gatsby'

import * as styles from './NotFound.module.scss'

const NotFoundPage = () => {
  return (
    <main className={styles.page}>
      <h1 className={styles.heading}>Page not found</h1>
      <p className={styles.paragraph}>
        Sorry 😔, we couldn’t find what you were looking for.
        <br />
        {process.env.NODE_ENV === 'development' ? (
          <>
            <br />
            Try creating a page in <code className={styles.code}>src/pages/</code>.
            <br />
          </>
        ) : null}
        <br />
        <Link to="/">Go home</Link>.
      </p>
    </main>
  )
}

export default NotFoundPage

export const Head = () => (
  <>
    <title>페이지를 찾을 수 없습니다 | Jiunbae's Blog</title>
    <meta name="robots" content="noindex, nofollow" />
    <meta name="description" content="요청하신 페이지를 찾을 수 없습니다." />
  </>
)
