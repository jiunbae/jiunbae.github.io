import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import playground from '@/data/playground';
import tools from '@/data/tools';

export const GET: APIRoute = async () => {
  const posts = await getCollection('posts', ({ data }) => data.published !== false);
  const notes = await getCollection('notes', ({ data }) => data.published !== false);

  const searchIndex = [
    ...posts.map(p => ({
      title: p.data.title,
      description: p.data.description || '',
      tags: p.data.tags || [],
      slug: `/posts${p.data.permalink || `/${p.slug}`}`,
      type: 'post' as const,
    })),
    ...notes.map(n => {
      const permalink = n.data.permalink || `/${n.slug}`;
      const slug = permalink.startsWith('/notes/')
        ? permalink
        : `/notes/${permalink.replace(/^\//, '')}`;
      return {
        title: n.data.title,
        description: n.data.description || '',
        tags: n.data.tags || [],
        slug,
        type: 'note' as const,
      };
    }),
    ...playground.map(p => ({
      title: p.title,
      description: p.description,
      tags: p.tags,
      slug: p.slug,
      type: 'playground' as const,
    })),
    ...tools.map(t => ({
      title: t.title,
      description: t.description,
      tags: t.tags || [],
      slug: t.slug,
      type: 'tool' as const,
    })),
  ];

  return new Response(JSON.stringify(searchIndex), {
    headers: { 'Content-Type': 'application/json' },
  });
};
