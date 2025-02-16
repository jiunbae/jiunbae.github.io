import React, { useEffect } from 'react'
import { useStaticQuery, graphql } from 'gatsby'

import { useTheme } from '@/contexts'

import * as styles from './Comments.module.scss'

export const Comments = () => {
  const { site: { siteMetadata : { repo } } } = useStaticQuery(graphql`
      query {
        site {
          siteMetadata {
            repo
          }
        }
      }
    `)
  const { theme } = useTheme()
  const commentsInjectionRoot: React.RefObject<HTMLDivElement | null> = React.createRef()

  useEffect(() => {
    if (commentsInjectionRoot.current === null) return
    if (commentsInjectionRoot.current.children.length === 0) {
      const scriptEl = document.createElement('script')
      scriptEl.setAttribute('src', 'https://utteranc.es/client.js')
      scriptEl.setAttribute('crossorigin', 'anonymous')
      scriptEl.setAttribute('async', 'true')
      scriptEl.setAttribute('repo', repo)
      scriptEl.setAttribute('issue-term', 'pathname')
      scriptEl.setAttribute('theme', `github-${theme}`)
      commentsInjectionRoot.current.appendChild(scriptEl)
    }
  }, [])

  useEffect(() => {
    if (document.querySelector('.utterances-frame')) {
      const iframe = document.querySelector<HTMLIFrameElement>('.utterances-frame')

      if (!iframe) {
        return
      }

      iframe?.contentWindow?.postMessage({ type: 'set-theme', theme: `github-${theme}` }, 'https://utteranc.es')
    }
  }, [theme])

  return (
    <div className={styles.comments}>
      <h1 className={styles.title}>
        Comments
      </h1>
      <hr className={styles.horizontalLine} />
      <div ref={commentsInjectionRoot} />
    </div>
  )
}
