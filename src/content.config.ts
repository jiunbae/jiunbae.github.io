import { glob } from 'astro/loaders';
import { defineCollection, z } from 'astro:content';

// Helper to handle null/undefined/empty values from YAML
const nullableString = () => z.string().nullish().transform(val => (val && val.trim()) ? val : undefined);

const baseSchema = z.object({
  title: z.string(),
  description: nullableString(),
  date: z.coerce.date(),
  permalink: nullableString(),
  tags: z.array(z.string()).default([]),
  published: z.boolean().default(true),
  heroImage: nullableString(),
  heroImageAlt: nullableString(),
});

// `[!_]` keeps the legacy convention of ignoring files prefixed with `_`
// (e.g. incidents/_example.md), which the Content Layer glob loader no longer
// excludes automatically.
const mdPattern = '**/[!_]*.{md,mdx}';

const posts = defineCollection({
  loader: glob({ pattern: mdPattern, base: './src/content/posts' }),
  schema: baseSchema,
});

const notes = defineCollection({
  loader: glob({ pattern: mdPattern, base: './src/content/notes' }),
  schema: baseSchema,
});

const reviews = defineCollection({
  loader: glob({ pattern: mdPattern, base: './src/content/reviews' }),
  schema: baseSchema.extend({
    rating: z.number().min(0).max(5).optional(),
  }),
});

const incidents = defineCollection({
  loader: glob({ pattern: mdPattern, base: './src/content/incidents' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    resolvedDate: z.coerce.date().optional(),
    severity: z.enum(['critical', 'major', 'minor', 'maintenance']),
    status: z.enum(['investigating', 'identified', 'monitoring', 'resolved', 'scheduled']),
    affectedServices: z.array(z.string()),
    published: z.boolean().default(true),
    timeline: z.array(z.object({
      time: z.coerce.date(),
      status: z.string(),
      message: z.string(),
    })).default([]),
  }),
});

export const collections = { posts, notes, reviews, incidents };
