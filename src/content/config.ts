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

export const collections = { posts, notes, reviews };
