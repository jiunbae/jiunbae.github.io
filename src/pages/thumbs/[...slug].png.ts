import type { APIRoute, GetStaticPaths } from 'astro';
import { getCollection } from 'astro:content';
import { Resvg } from '@resvg/resvg-js';
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const fontFiles: string[] = [];
const fontBuffers: Buffer[] = [];

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

const THUMB_WIDTH = 1200;
const THUMB_HEIGHT = 675;

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

const fontFamily = "'Noto Sans KR', 'Noto Sans CJK KR', sans-serif";

const createThumbSvg = (title: string, siteName: string) => {
  const titleLines = wrapText(title, 20, 3);
  const padding = 90;
  const titleFontSize = 68;
  const titleLineHeight = 86;
  const titleStartY = 180;

  const titleSpans = titleLines
    .map((line, i) => `<tspan x="${padding}" dy="${i === 0 ? 0 : titleLineHeight}">${escapeXml(line)}</tspan>`)
    .join('');

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${THUMB_WIDTH}" height="${THUMB_HEIGHT}" viewBox="0 0 ${THUMB_WIDTH} ${THUMB_HEIGHT}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="thumb-bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1e293b" />
    </linearGradient>
  </defs>
  <rect fill="url(#thumb-bg)" width="${THUMB_WIDTH}" height="${THUMB_HEIGHT}" rx="40" />
  <text x="${padding}" y="${titleStartY}" fill="#f8fafc" font-family="${fontFamily}" font-size="${titleFontSize}" font-weight="700">
    ${titleSpans}
  </text>
  <text x="${padding}" y="${THUMB_HEIGHT - 70}" fill="rgba(248, 250, 252, 0.7)" font-family="${fontFamily}" font-size="36" font-weight="400">${escapeXml(siteName)}</text>
</svg>`;
};

export const getStaticPaths: GetStaticPaths = async () => {
  const posts = await getCollection('posts', ({ data }) => data.published !== false);

  return posts.map(p => {
    let postSlug = p.data.permalink || p.slug;
    if (postSlug.startsWith('/')) postSlug = postSlug.slice(1);
    return {
      params: { slug: `posts/${postSlug}` },
      props: {
        title: p.data.title,
        siteName: 'jiun.dev',
      },
    };
  });
};

export const GET: APIRoute = async ({ props }) => {
  const { title, siteName } = props as { title: string; siteName: string };

  const svg = createThumbSvg(title, siteName);

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
    fitTo: { mode: 'width', value: THUMB_WIDTH },
    font: fontOptions as Record<string, unknown>,
    languages: ['en', 'ko'],
  });
  const png = resvg.render().asPng();

  return new Response(png, {
    headers: { 'Content-Type': 'image/png' },
  });
};
