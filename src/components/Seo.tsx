import { graphql, useStaticQuery } from 'gatsby'
import { PropsWithChildren } from 'react'

import { useTheme } from '@/contexts'

interface SeoProps {
  title?: string
  description?: string
  heroImage?: string
  pathname: string
  publishedTime?: string
  modifiedTime?: string
  author?: string
  tags?: string[]
  type?: 'website' | 'article'
}

interface SeoQuery {
  file: {
    publicURL: string
  }
  site: {
    siteMetadata: {
      title: string
      description: string
      siteUrl: string
    }
  }
}

export const Seo = ({
  title,
  description,
  heroImage,
  pathname,
  publishedTime,
  modifiedTime,
  author,
  tags,
  type,
  children
}: PropsWithChildren<SeoProps>) => {
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

  const resolvedTitle = title || defaultTitle
  const resolvedDescription = description || defaultDescription
  const isAbsolutePath = /^https?:\/\//i.test(pathname)
  const url = isAbsolutePath ? pathname : `${siteUrl}${pathname || ''}`
  const isHeroAbsolute = heroImage ? /^https?:\/\//i.test(heroImage) : false
  const image = heroImage
    ? (isHeroAbsolute ? heroImage : `${siteUrl}${heroImage}`)
    : `${siteUrl}${defaultImage}`

  const ogType = type === 'article' || publishedTime ? 'article' : 'website'
  const siteName = defaultTitle

  const seo = {
    title: resolvedTitle,
    description: resolvedDescription,
    url,
    image
  }

  return (
    <>
      <title>{seo.title}</title>
      <link rel="canonical" href={seo.url} />
      <meta name="description" content={seo.description} />
      <meta name="image" content={seo.image} />
      <meta name="theme-color" content={theme == 'dark' ? '#242424' : '#f4f4f4fa'} />
      {/* Open Graph / Facebook */}
      <meta property="og:locale" content="ko_KR" />
      <meta property="og:site_name" content={siteName} />
      <meta property="og:title" content={seo.title} />
      <meta property="og:description" content={seo.description} />
      <meta property="og:type" content={ogType} />
      <meta property="og:url" content={seo.url} />
      <meta property="og:image" content={seo.image} />
      {publishedTime ? <meta property="article:published_time" content={publishedTime} /> : null}
      {modifiedTime ? <meta property="article:modified_time" content={modifiedTime} /> : null}
      {author ? <meta property="article:author" content={author} /> : null}
      {tags?.map((tag, index) => (
        <meta property="article:tag" content={tag} key={`article-tag-${index}`} />
      ))}
      {/* Twitter */}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={seo.title} />
      <meta name="twitter:description" content={seo.description} />
      <meta name="twitter:image" content={seo.image} />
      {children}
    </>
  )
}
