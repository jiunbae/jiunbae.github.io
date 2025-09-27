import type { GatsbyConfig } from 'gatsby'
import path from 'path'

const config: GatsbyConfig = {
  siteMetadata: {
    title: 'Jiunbae\'s Blog',
    heading: 'Jiunbae\'s',
    description: 'Jiunbae\'s Blog',
    name: {
      kr: '배지운',
      en: 'Jiun Bae'
    },
    siteUrl: 'https://blog.jiun.dev',
    keywords: ['jiunbae', 'blog', 'dev'],
    heroImage: path.resolve(__dirname, 'src/images/cover.png'),
    repo: 'jiunbae/jiunbae.github.io',
    social: {
      email: 'mailto:jiunbae.623@gmail.com',
      facebook: 'https://www.facebook.com/MayTryArk',
      linkedin: 'https://linkedin.com/in/jiunbae',
      github: 'https://github.com/jiunbae',
      twitter: 'https://twitter.com/baejiun',
      instagram: 'https://instagram.com/bae.jiun'
    }
  },
  graphqlTypegen: true,
  jsxRuntime: 'automatic',
  plugins: [
    {
      resolve: 'gatsby-plugin-google-gtag',
      options: {
        trackingIds: ['G-8JQSR962RQ'],
        pluginConfig: {
          head: true
        }
      }
    },
    {
      resolve: 'gatsby-plugin-sass',
      options: {
        sassOptions: {
          api: 'modern-compiler'
        }
      }
    },
    'gatsby-plugin-advanced-sitemap-v5',
    'gatsby-plugin-image',
    'gatsby-plugin-sharp',
    'gatsby-transformer-sharp',
    {
      resolve: 'gatsby-transformer-remark',
      options: {
        plugins: [
          {
            resolve: 'gatsby-remark-images',
            options: {
              maxWidth: 780,
              linkImagesToOriginal: false,
              wrapperStyle: 'border-radius: 5px; overflow: hidden;'
            }
          },
          {
            resolve: 'gatsby-remark-autolink-headers',
            options: {
              icon: false
            }
          },
          {
            resolve: 'gatsby-remark-prismjs',
            options: {
              inlineCodeMarker: '>'
            }
          }
        ]
      }
    },
    {
      resolve: 'gatsby-plugin-react-svg',
      options: {
        rule: {
          include: /src\/images/,
          options: {
            props: {
              className: 'my-class'
            }
          }
        }
      }
    },
    {
      resolve: 'gatsby-plugin-manifest',
      options: {
        icon: path.resolve(__dirname, 'static/profile.png')
      }
    },
    {
      resolve: 'gatsby-source-filesystem',
      options: {
        name: 'contents',
        path: path.resolve(__dirname, 'contents'),
        ignore: ['**/notes/**']
      }
    },
    {
      resolve: 'gatsby-source-filesystem',
      options: {
        name: 'notes',
        path: path.resolve(__dirname, 'contents', 'notes')
      }
    },
    {
      resolve: 'gatsby-source-filesystem',
      options: {
        name: 'images',
        path: path.resolve(__dirname, 'src/images')
      }
    },
    {
      resolve: 'gatsby-plugin-feed',
      options: {
        query: `
          {
            site {
              siteMetadata {
                title
                description
                siteUrl
                site_url: siteUrl
              }
            }
          }
        `,
        feeds: [
          {
            serialize: ({ query: { site, allMarkdownRemark } }: any) => {
              return allMarkdownRemark.nodes.map((node: any) => {
                const description = node.frontmatter.description ?? node.excerpt ?? ''
                return Object.assign({}, node.frontmatter, {
                  description,
                  date: new Date(node.frontmatter.date),
                  url: `${site.siteMetadata.siteUrl}/posts${node.frontmatter.slug}`,
                  guid: `${site.siteMetadata.siteUrl}/posts${node.frontmatter.slug}`,
                  custom_elements: [{ 'content:encoded': node.html }]
                })
              })
            },
            query: `
              {
                allMarkdownRemark(sort: { frontmatter: { date: DESC }}) {
                  nodes {
                    frontmatter {
                      date
                      description
                      slug
                      title
                    }
                    excerpt(pruneLength: 200)
                    html
                  }
                }
              }
            `,
            output: '/rss.xml',
            title: 'jiunbae blog RSS feed'
          }
        ]
      }
    },
    {
      resolve: 'gatsby-transformer-json',
      options: {
        typeName: 'Json'
      }
    }
  ]
}

export default config
