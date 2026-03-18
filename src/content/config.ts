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

const posts = defineCollection({
  type: 'content',
  schema: baseSchema,
});

const notes = defineCollection({
  type: 'content',
  schema: baseSchema,
});

const reviews = defineCollection({
  type: 'content',
  schema: baseSchema.extend({
    rating: z.number().min(0).max(5).optional(),
  }),
});

const incidents = defineCollection({
  type: 'content',
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
