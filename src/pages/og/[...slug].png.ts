import type { APIRoute, GetStaticPaths } from 'astro';
import { getCollection } from 'astro:content';
import { Resvg } from '@resvg/resvg-js';
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

// Load font files for Resvg rendering
const fontFiles: string[] = [];
const fontBuffers: Buffer[] = [];

// Try bundled TTF first (static/fonts), then fallback to @fontsource woff
const ttfPaths = [
  join(process.cwd(), 'public', 'fonts', 'noto-sans-kr-korean-700-normal.ttf'),
  join(process.cwd(), 'public', 'fonts', 'noto-sans-kr-korean-400-normal.ttf'),
];

const woffPaths = [
  join(process.cwd(), 'node_modules/@fontsource/noto-sans-kr/files/noto-sans-kr-korean-700-normal.woff'),
  join(process.cwd(), 'node_modules/@fontsource/noto-sans-kr/files/noto-sans-kr-korean-400-normal.woff'),
];

for (const p of ttfPaths) {
  if (existsSync(p)) fontFiles.push(p);
}

if (fontFiles.length === 0) {
  for (const p of woffPaths) {
    if (existsSync(p)) fontBuffers.push(readFileSync(p));
  }
}

const OG_WIDTH = 1200;
const OG_HEIGHT = 630;

const escapeXml = (value: string) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

const wrapText = (text: string, maxCharsPerLine: number, maxLines: number) => {
  const words = text.replace(/\s+/g, ' ').trim().split(' ');
  const lines: string[] = [];
  let currentLine = '';
  let remaining = false;

  for (const word of words) {
    const nextLine = currentLine.length === 0 ? word : `${currentLine} ${word}`;
    if (nextLine.length > maxCharsPerLine && currentLine.length > 0) {
      lines.push(currentLine);
      currentLine = word;
    } else {
      currentLine = nextLine;
    }
    if (lines.length === maxLines) {
      remaining = true;
      break;
    }
  }

  if (lines.length < maxLines && currentLine) {
    lines.push(currentLine);
  }

  // Add ellipsis if text was truncated
  if (remaining || lines.length > maxLines) {
    lines.length = Math.min(lines.length, maxLines);
    const lastLine = lines[maxLines - 1];
    if (lastLine.length > maxCharsPerLine - 1) {
      lines[maxLines - 1] = `${lastLine.slice(0, maxCharsPerLine - 1).trim()}…`;
    } else {
      lines[maxLines - 1] = `${lastLine}…`;
    }
  }

  return lines;
};

const truncateSummary = (text: string, maxChars = 180, maxLines = 3) => {
  const normalized = text.replace(/\s+/g, ' ').trim();
  const truncated = normalized.length > maxChars ? `${normalized.slice(0, maxChars - 1).trim()}…` : normalized;
  return wrapText(truncated, 28, maxLines);
};

const fontFamily = "'Noto Sans KR', 'Noto Sans CJK KR', sans-serif";

const createOgSvg = (
  title: string,
  summary: string,
  date: string,
  siteName: string,
) => {
  const titleLines = wrapText(title, 18, 3);
  const summaryLines = truncateSummary(summary);
  const padding = 120;
  const titleFontSize = 52;
  const titleLineHeight = 66;
  const summaryFontSize = 36;
  const summaryLineHeight = 50;

  const titleStartY = 130;
  const summaryStartY = titleStartY + titleLines.length * titleLineHeight + 48;

  const titleSpans = titleLines
    .map((line, i) => `<tspan x="${padding}" dy="${i === 0 ? 0 : titleLineHeight}">${escapeXml(line)}</tspan>`)
    .join('');

  const summarySpans = summaryLines
    .map((line, i) => `<tspan x="${padding}" dy="${i === 0 ? 0 : summaryLineHeight}">${escapeXml(line)}</tspan>`)
    .join('');

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${OG_WIDTH}" height="${OG_HEIGHT}" viewBox="0 0 ${OG_WIDTH} ${OG_HEIGHT}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="og-bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1e293b" />
    </linearGradient>
  </defs>
  <rect fill="url(#og-bg)" width="${OG_WIDTH}" height="${OG_HEIGHT}" rx="32" />
  <text x="${padding}" y="${titleStartY}" fill="#f8fafc" font-family="${fontFamily}" font-size="${titleFontSize}" font-weight="700">
    ${titleSpans}
  </text>
  <text x="${padding}" y="${summaryStartY}" fill="rgba(248, 250, 252, 0.9)" font-family="${fontFamily}" font-size="${summaryFontSize}" font-weight="400">
    ${summarySpans}
  </text>
  <g font-family="${fontFamily}" font-size="32" font-weight="400" fill="rgba(248, 250, 252, 0.68)">
    <text x="${padding}" y="555">${escapeXml(siteName)}</text>
    <text x="${OG_WIDTH - padding}" y="555" text-anchor="end">${escapeXml(date)}</text>
  </g>
</svg>`;
};

export const getStaticPaths: GetStaticPaths = async () => {
  const posts = await getCollection('posts', ({ data }) => data.published !== false);
  const notes = await getCollection('notes', ({ data }) => data.published !== false);
  const reviews = await getCollection('reviews', ({ data }) => data.published !== false);

  const paths = [
    ...posts.map(p => {
      let postSlug = p.data.permalink || p.slug;
      if (postSlug.startsWith('/')) postSlug = postSlug.slice(1);
      return {
        params: { slug: `posts/${postSlug}` },
        props: {
          title: p.data.title,
          description: p.data.description || '',
          date: p.data.date.toISOString().split('T')[0].replace(/-/g, '.'),
          siteName: 'jiun.dev',
        },
      };
    }),
    ...notes.map(n => {
      let noteSlug = n.data.permalink || n.slug;
      if (noteSlug.startsWith('/notes/')) noteSlug = noteSlug.slice(7);
      else if (noteSlug.startsWith('/')) noteSlug = noteSlug.slice(1);
      return {
        params: { slug: `notes/${noteSlug}` },
        props: {
          title: n.data.title,
          description: n.data.description || '',
          date: n.data.date.toISOString().split('T')[0].replace(/-/g, '.'),
          siteName: 'notes.jiun.dev',
        },
      };
    }),
    ...reviews.map(r => {
      let reviewSlug = r.data.permalink || r.slug;
      if (reviewSlug.startsWith('/reviews/')) reviewSlug = reviewSlug.slice(9);
      else if (reviewSlug.startsWith('/')) reviewSlug = reviewSlug.slice(1);
      return {
        params: { slug: `reviews/${reviewSlug}` },
        props: {
          title: r.data.title,
          description: r.data.description || '',
          date: r.data.date.toISOString().split('T')[0].replace(/-/g, '.'),
          siteName: 'jiun.dev',
        },
      };
    }),
  ];

  return paths;
};

export const GET: APIRoute = async ({ props }) => {
  const { title, description, date, siteName } = props as {
    title: string;
    description: string;
    date: string;
    siteName: string;
  };

  const svg = createOgSvg(title, description, date, siteName);

  interface ResvgFontOptions {
    loadSystemFonts: boolean;
    defaultFontFamily: string;
    sansSerifFamily: string;
    fontBuffers?: Buffer[];
    fontFiles?: string[];
  }

  const fontOptions: ResvgFontOptions = {
    loadSystemFonts: false,
    defaultFontFamily: 'Noto Sans KR',
    sansSerifFamily: 'Noto Sans KR',
  };

  if (fontBuffers.length > 0) {
    fontOptions.fontBuffers = fontBuffers;
  } else if (fontFiles.length > 0) {
    fontOptions.fontFiles = fontFiles;
  }

  const resvg = new Resvg(svg, {
    fitTo: { mode: 'width', value: OG_WIDTH },
    font: fontOptions as Record<string, unknown>,
    languages: ['en', 'ko'],
  });
  const png = resvg.render().asPng();

  return new Response(png, {
    headers: { 'Content-Type': 'image/png' },
  });
};
