import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import sitemap from '@astrojs/sitemap';
import expressiveCode from 'astro-expressive-code';
import { unified } from '@astrojs/markdown-remark';
import rehypeMermaid from './src/plugins/rehype-mermaid.mjs';

// https://astro.build/config
export default defineConfig({
  site: 'https://jiun.dev',
  // Astro 7부터 compressHTML 기본값이 'jsx'로 바뀌어 인라인 요소 사이 공백을
  // JSX 규칙대로 제거한다. 본문에 인라인 링크·강조가 섞인 한국어 문단이 많아
  // 공백이 사라지면 눈에 띄므로 Astro 6과 같은 HTML 인식 압축을 유지한다.
  compressHTML: true,
  redirects: {
    // kiwimu 글이 v1.0 페르소나 리뷰에서 전체 여정(v1.2)으로 확장되며 슬러그 변경.
    // 기존에 공유·색인된 링크가 깨지지 않도록 리다이렉트 유지.
    '/posts/kiwimu-14-persona-review': '/posts/kiwimu',
    // 이게모야 페이지는 igemoya.jiun.dev 로 분리했다. App Store Connect 초안 등
    // 이미 이 경로를 가리키는 링크가 404가 되지 않도록 남겨 둔다.
    '/apps/igemoya': 'https://igemoya.jiun.dev/',
    '/apps/igemoya/privacy': 'https://igemoya.jiun.dev/privacy/',
    '/apps/igemoya/terms': 'https://igemoya.jiun.dev/terms/',
    '/apps/igemoya/support': 'https://igemoya.jiun.dev/support/',
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
    // Astro 7은 Sätteri를 기본 마크다운 프로세서로 쓴다. rehype 플러그인은
    // remark/rehype 파이프라인 전용이므로 프로세서를 명시해야 계속 동작한다.
    // 최상위 markdown.rehypePlugins는 deprecated라 unified()에 직접 넘긴다.
    processor: unified({ rehypePlugins: [rehypeMermaid] }),
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
