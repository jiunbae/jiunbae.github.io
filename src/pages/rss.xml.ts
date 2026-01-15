import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
  const posts = await getCollection('posts', ({ data }) => data.published !== false);
  const sortedPosts = posts.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());

  return rss({
    title: "Jiunbae's Blog",
    description: "AI/ML 연구원의 기술 블로그. 딥러닝, DevOps, Home Lab 실전 경험을 공유합니다.",
    site: context.site!,
    items: sortedPosts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.date,
      description: post.data.description || '',
      link: `/posts${post.data.permalink}`,
    })),
    customData: `<language>ko</language>`,
  });
}
