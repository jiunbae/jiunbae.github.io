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
  setHeadComponents([
    <style
      key="theme-transition-style"
      dangerouslySetInnerHTML={{
        __html: `.${'theme-transition-prevent'} *, .${'theme-transition-prevent'} *::before, .${'theme-transition-prevent'} *::after { transition: none !important; }`
      }}
    />,
    <script key="theme-initializer" dangerouslySetInnerHTML={{ __html: ThemeInitializer }} />
  ])
}
