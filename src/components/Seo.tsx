import { graphql, useStaticQuery } from 'gatsby'
import { PropsWithChildren } from 'react'

import { useTheme } from '@/contexts'

interface SeoProps {
  title?: string;
  description?: string;
  heroImage?: string;
  pathname: string;
};

interface SeoQuery {
  file: {
    publicURL: string;
  };
  site: {
    siteMetadata: {
      title: string;
      description: string;
      siteUrl: string;
    };
  };
};

export const Seo = ({ title, description, heroImage, pathname, children }: PropsWithChildren<SeoProps>) => {
  const { theme } = useTheme()
  const data = useStaticQuery<SeoQuery>(graphql`
    query SeoQuery {
      site {
        siteMetadata {
          title
          description
          siteUrl
        }
      }
      file(relativePath: { eq: "cover.png" }) {
        publicURL
      }
    }
  `)

  const { title: defaultTitle, description: defaultDescription, siteUrl } = data.site.siteMetadata
  const { publicURL: defaultImage } = data.file

  const seo = {
    title: title || defaultTitle,
    description: description || defaultDescription,
    url: `${siteUrl}${pathname || ''}`,
    image: `${siteUrl}${heroImage || defaultImage}`
  }

  return (
    <>
      <title>{seo.title}</title>
      <link rel="canonical" href={seo.url} />
      <meta name="description" content={seo.description} />
      <meta name="image" content={seo.image} />
      <meta name="theme-color" content={theme == 'dark' ? '#242424' : '#f4f4f4fa'}/>
      {/* Open Graph / Facebook */}
      <meta property="og:title" content={seo.title} />
      <meta property="og:description" content={seo.description} />
      <meta property="og:type" content="blog" />
      <meta property="og:url" content={seo.url}></meta>
      <meta property="og:image" content={seo.image}></meta>
      {/* Twitter */}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={seo.title} />
      <meta name="twitter:description" content={seo.description} />
      <meta property="twitter:image" content={seo.image}></meta>
      {children}
    </>
  )
}
