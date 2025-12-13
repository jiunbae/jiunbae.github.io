export const sanitizeSlug = (slug: string, fallback = 'page') => {
  const trimmed = slug.replace(/^\/+/, '').replace(/\/+$/, '')
  return trimmed.replace(/[^a-zA-Z0-9-_]/g, '-').replace(/-+/g, '-').toLowerCase() || fallback
}

export const sanitizeNoteSlug = (slug: string) => sanitizeSlug(slug, 'note')
export const sanitizePostSlug = (slug: string) => sanitizeSlug(slug, 'post')
export const sanitizeReviewSlug = (slug: string) => sanitizeSlug(slug, 'review')
