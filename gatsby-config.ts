import type { GatsbyConfig } from "gatsby";
const path = require("path");

const config: GatsbyConfig = {
  siteMetadata: {
    title: `Jiunbae's Blog`,
    description: `Jiunbae's Blog`,
    siteUrl: `https://blog.jiun.dev`,
    keywords: ["jiunbae", "blog", "dev"],
    heroImage: path.resolve(__dirname, "src/images/cover.png"),
  },
  graphqlTypegen: true,
  jsxRuntime: "automatic",
  plugins: [
    {
      resolve: `gatsby-plugin-google-gtag`,
      options: {
        trackingIds: ["G-8JQSR962RQ"],
        pluginConfig: {
          head: true,
        },
      },
    },
    "gatsby-plugin-sass",
    `gatsby-plugin-advanced-sitemap-v5`,
    `gatsby-plugin-image`,
    `gatsby-plugin-sharp`,
    `gatsby-transformer-sharp`,
    {
      resolve: `gatsby-transformer-remark`,
      options: {
        plugins: [
          {
            resolve: `gatsby-remark-images`,
            options: {
              maxWidth: 780,
              linkImagesToOriginal: false,
              wrapperStyle: "border-radius: 5px; overflow: hidden;",
            },
          },
          {
            resolve: `gatsby-remark-autolink-headers`,
            options: {
              icon: false,
            },
          },
          {
            resolve: `gatsby-remark-prismjs`,
            options: {
              inlineCodeMarker: `>`,
            },
          },
        ],
      },
    },
    {
      resolve: "gatsby-plugin-react-svg",
      options: {
        rule: {
          include: /src\/images/,
          options: {
            props: {
              className: "my-class",
            },
          },
        },
      },
    },
    {
      resolve: `gatsby-plugin-manifest`,
      options: {
        icon: path.resolve(__dirname, "static/favicon.ico"),
      },
    },
    {
      resolve: "gatsby-source-filesystem",
      options: {
        name: "contents",
        path: path.resolve(__dirname, "contents"),
      },
    },
    {
      resolve: "gatsby-source-filesystem",
      options: {
        name: "images",
        path: path.resolve(__dirname, "src/images"),
      },
    },
    {
      resolve: `gatsby-plugin-feed`,
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
                return Object.assign({}, node.frontmatter, {
                  description: node.frontmatter.description,
                  date: new Date(node.frontmatter.date),
                  url: `${site.siteMetadata.siteUrl}/posts${node.frontmatter.slug}`,
                  guid: `${site.siteMetadata.siteUrl}/posts${node.frontmatter.slug}`,
                  custom_elements: [{ "content:encoded": node.html }],
                });
              });
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
                    html
                  }
                }
              }
            `,
            output: "/rss.xml",
            title: "jiunbae blog RSS feed",
          },
        ],
      },
    },
  ],
};

export default config;
