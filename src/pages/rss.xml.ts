import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
  const posts = await getCollection('posts', ({ data }) => data.published !== false);
  const notes = await getCollection('notes', ({ data }) => data.published !== false);

  const postItems = posts.map((post) => ({
    title: post.data.title,
    pubDate: post.data.date,
    description: post.data.description || '',
    link: `/posts${post.data.permalink}`,
  }));

  const noteItems = notes.map((note) => ({
    title: `[Note] ${note.data.title}`,
    pubDate: note.data.date,
    description: note.data.description || '',
    link: `/notes/${note.data.permalink || note.slug}/`,
  }));

  const allItems = [...postItems, ...noteItems].sort(
    (a, b) => b.pubDate.valueOf() - a.pubDate.valueOf()
  );

  return rss({
    title: "Jiunbae's Blog",
    description: "AI/ML 연구원의 기술 블로그. 딥러닝, DevOps, Home Lab 실전 경험을 공유합니다.",
    site: context.site!,
    items: allItems,
    customData: `<language>ko</language>`,
  });
}
