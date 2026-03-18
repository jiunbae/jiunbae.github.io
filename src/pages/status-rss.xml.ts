import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

const severityPrefix: Record<string, string> = {
  critical: '🔴',
  major: '🟠',
  minor: '🟡',
  maintenance: '🔧',
};

export async function GET(context: APIContext) {
  const incidents = await getCollection('incidents', ({ data }) => data.published !== false);

  const items = incidents
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf())
    .map((incident) => ({
      title: `${severityPrefix[incident.data.severity] || ''} ${incident.data.title}`,
      pubDate: incident.data.date,
      description: `[${incident.data.status}] Affected: ${incident.data.affectedServices.join(', ')}`,
      link: `/status/${incident.slug}/`,
    }));

  return rss({
    title: "jiun.dev — Status",
    description: "Service status and incident reports for jiun.dev services.",
    site: context.site!,
    items,
    customData: `<language>ko</language>`,
  });
}
