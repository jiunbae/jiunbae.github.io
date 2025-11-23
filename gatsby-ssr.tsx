import type { GatsbySSR } from 'gatsby'

import { ThemeProvider } from './src/contexts'
import Layout from './src/layouts'

const ThemeInitializer = `(() => {
  const storageKey = 'theme'
  const className = 'theme-transition-prevent'

  const root = document.documentElement
  if (!root) return

  const disableTransitions = () => {
    root.classList.add(className)

    return () => {
      root.classList.remove(className)
    }
  }

  const enableTransitions = disableTransitions()

  const getPreferredTheme = () => {
    try {
      const stored = window.localStorage.getItem(storageKey)
      if (stored === 'dark' || stored === 'light') {
        return stored
      }
    } catch {
      // ignore read errors
    }

    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }

  const theme = getPreferredTheme()
  root.setAttribute('data-theme', theme)
  root.style.setProperty('color-scheme', theme)

  window.requestAnimationFrame(() => {
    enableTransitions()
  })
})()
`

export const wrapPageElement: GatsbySSR['wrapPageElement'] = ({ element, props }) => {
  const pathname = props.location.pathname

  return (
    <ThemeProvider>
      <Layout pathname={pathname}>{element}</Layout>
    </ThemeProvider>
  )
}

export const onRenderBody: GatsbySSR['onRenderBody'] = ({ setHeadComponents }) => {
  const components = [
    <style
      key="theme-transition-style"
      dangerouslySetInnerHTML={{
        __html: `.${'theme-transition-prevent'} *, .${'theme-transition-prevent'} *::before, .${'theme-transition-prevent'} *::after { transition: none !important; }`
      }}
    />,
    <script key="theme-initializer" dangerouslySetInnerHTML={{ __html: ThemeInitializer }} />
  ]

  // Content Security Policy는 프로덕션 빌드에서만 적용
  if (process.env.NODE_ENV === 'production') {
    const cspContent = `
      default-src 'self';
      script-src 'self' 'unsafe-inline' https://www.googletagmanager.com https://www.google-analytics.com;
      style-src 'self' 'unsafe-inline';
      img-src 'self' data: https: blob:;
      font-src 'self' data:;
      connect-src 'self' https://api.github.com https://www.google-analytics.com https://www.googletagmanager.com;
      frame-ancestors 'none';
      base-uri 'self';
      form-action 'self';
      object-src 'none';
    `.replace(/\s+/g, ' ').trim()

    components.unshift(
      <meta
        key="csp"
        httpEquiv="Content-Security-Policy"
        content={cspContent}
      />
    )
  }

  setHeadComponents(components)
}
