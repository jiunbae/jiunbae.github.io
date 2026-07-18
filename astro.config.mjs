import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import sitemap from '@astrojs/sitemap';
import expressiveCode from 'astro-expressive-code';
import rehypeMermaid from './src/plugins/rehype-mermaid.mjs';

// https://astro.build/config
export default defineConfig({
  site: 'https://jiun.dev',
  redirects: {
    // kiwimu 글이 v1.0 페르소나 리뷰에서 전체 여정(v1.2)으로 확장되며 슬러그 변경.
    // 기존에 공유·색인된 링크가 깨지지 않도록 리다이렉트 유지.
    '/posts/kiwimu-14-persona-review': '/posts/kiwimu',
  },
  integrations: [
    expressiveCode({
      themes: ['github-dark', 'github-light'],
      styleOverrides: {
        borderRadius: '0.5rem',
        frames: {
          frameBoxShadowCssValue: 'none',
          terminalTitlebarDotsOpacity: '0',
          terminalTitlebarBackground: 'transparent',
          terminalTitlebarBorderBottomColor: 'transparent',
          editorTabBarBackground: 'transparent',
          editorTabBarBorderBottomColor: 'transparent',
        },
        codeBackground: 'var(--code-bg, #1e1e1e)',
      },
    }),
    mdx(),
    react(),
    sitemap(),
  ],
  markdown: {
    rehypePlugins: [rehypeMermaid],
  },
  vite: {
    css: {
      preprocessorOptions: {
        scss: {
          api: 'modern-compiler',
        },
      },
    },
  },
});
