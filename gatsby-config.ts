import type { GatsbyConfig } from 'gatsby'
import path from 'path'

// RSS Feed 타입 정의
type FeedQuery = {
  site: {
    siteMetadata: {
      siteUrl: string
    }
  }
  allMarkdownRemark: {
    nodes: Array<{
      frontmatter: {
        date: string
        description?: string | null
        slug: string
        title: string
      }
      excerpt?: string | null
      html: string
    }>
  }
}

const config: GatsbyConfig = {
  siteMetadata: {
    title: 'Jiunbae\'s Blog',
    heading: 'Jiunbae\'s',
    description: 'AI/ML 연구원의 기술 블로그. 딥러닝, DevOps, Home Lab 실전 경험을 공유합니다.',
    name: {
      kr: '배지운',
      en: 'Jiun Bae'
    },
    author: {
      name: 'Jiun Bae',
      nameKr: '배지운',
      email: 'jiunbae.623@gmail.com'
    },
    language: 'ko',
    siteUrl: 'https://jiun.dev',
    keywords: ['AI', '인공지능', 'ML', '머신러닝', 'DevOps', '데브옵스', 'Home Lab', '홈랩', 'LLM', '거대언어모델', 'Claude', 'Kubernetes', '쿠버네티스', '기술블로그'],
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
  flags: {
    // 성능 최적화 플래그
    FAST_DEV: true,
  },
  plugins: [
    // 번들 사이즈 분석 (빌드 시에만 활성화)
    ...(process.env.ANALYZE_BUNDLE === 'true' ? [{
      resolve: 'gatsby-plugin-webpack-bundle-analyser-v2',
      options: {
        devMode: false,
        analyzerMode: 'static',
        reportFilename: 'bundle-report.html',
        openAnalyzer: false,
      }
    }] : []),
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
            resolve: 'gatsby-remark-responsive-iframe',
            options: {
              wrapperStyle: 'margin-bottom: 1rem'
            }
          },
          {
            resolve: 'gatsby-remark-autolink-headers',
            options: {
              icon: false
            }
          },
          {
            resolve: 'gatsby-remark-mermaid',
            options: {
              mermaidConfig: {
                theme: 'default',
                themeVariables: {
                  fontFamily: 'inherit'
                }
              }
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
    // {
    //   resolve: 'gatsby-plugin-manifest',
    //   options: {
    //     icon: path.resolve(__dirname, 'static/profile.png')
    //   }
    // },
    {
      resolve: 'gatsby-source-filesystem',
      options: {
        name: 'posts',
        path: path.resolve(__dirname, 'contents', 'posts'),
        ignore: ['**/images/**', '**/README.md']
      }
    },
    {
      resolve: 'gatsby-source-filesystem',
      options: {
        name: 'data',
        path: path.resolve(__dirname, 'contents', 'data')
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
        name: 'reviews',
        path: path.resolve(__dirname, 'contents', 'reviews')
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
            serialize: ({ query }: { query: FeedQuery }) => {
              const { site, allMarkdownRemark } = query
              return allMarkdownRemark.nodes.map((node) => {
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
