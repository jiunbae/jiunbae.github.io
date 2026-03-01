import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import sitemap from '@astrojs/sitemap';
import expressiveCode from 'astro-expressive-code';
import rehypeMermaid from './src/plugins/rehype-mermaid.mjs';

// https://astro.build/config
export default defineConfig({
  site: 'https://jiun.dev',
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
