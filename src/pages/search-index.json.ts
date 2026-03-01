import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';

export const GET: APIRoute = async () => {
  const posts = await getCollection('posts', ({ data }) => data.published !== false);
  const notes = await getCollection('notes', ({ data }) => data.published !== false);

  const searchIndex = [
    ...posts.map(p => ({
      title: p.data.title,
      description: p.data.description || '',
      tags: p.data.tags || [],
      slug: `/posts/${p.data.permalink || p.slug}/`,
      type: 'post' as const,
    })),
    ...notes.map(n => ({
      title: n.data.title,
      description: n.data.description || '',
      tags: n.data.tags || [],
      slug: `/notes/${n.data.permalink || n.slug}/`,
      type: 'note' as const,
    })),
  ];

  return new Response(JSON.stringify(searchIndex), {
    headers: { 'Content-Type': 'application/json' },
  });
};
