import type { PropsWithChildren } from 'react'

import { Footer, Header, SkipLink } from './components'

type LayoutProps = PropsWithChildren<{ pathname: string }>;

const Layout = ({ pathname, children }: LayoutProps) => {
  return (
    <>
      <SkipLink />
      <Header pathname={pathname} />
      <main id="main-content">
        {children}
      </main>
      <Footer />
    </>
  )
}

export default Layout
