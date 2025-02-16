import 'prismjs/themes/prism-tomorrow.css'
import '@/styles/index.scss'

import type { WrapPageElementBrowserArgs } from 'gatsby'

import { ThemeProvider } from '@/contexts'
import Layout from '@/layouts'

export const wrapPageElement = ({ element, props }: WrapPageElementBrowserArgs) => {
  const pathname = props.location.pathname

  return (
    <ThemeProvider>
      <Layout pathname={pathname}>{element}</Layout>
    </ThemeProvider>
  )
}
