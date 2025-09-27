const NOTE_ANCHOR_PREFIX = 'note-'

export const sanitizeNoteSlug = (slug: string) => {
  const trimmed = slug.replace(/^\/+/, '').replace(/\/+$/, '')
  const normalized = trimmed.replace(/[^a-zA-Z0-9-_]/g, '-').replace(/-+/g, '-').toLowerCase()
  return normalized || 'note'
}

export const getNoteAnchorId = (slug: string) => `${NOTE_ANCHOR_PREFIX}${sanitizeNoteSlug(slug)}`

export const getNotePagePath = (slug: string) => `/notes/${sanitizeNoteSlug(slug)}/`

export const buildNoteShareUrl = (slug: string) => {
  if (typeof window !== 'undefined') {
    const url = new URL(getNotePagePath(slug), window.location.origin)
    return url.toString()
  }

  // Fallback during SSR - return absolute path string only
  return getNotePagePath(slug)
}

export const extractSlugFromLocation = (search: string, hash: string) => {
  const searchParams = new URLSearchParams(search)
  const fromQuery = searchParams.get('note')
  if (fromQuery) {
    return sanitizeNoteSlug(fromQuery)
  }

  if (hash.startsWith('#') && hash.length > 1) {
    return sanitizeNoteSlug(decodeURIComponent(hash.substring(1)))
  }

  return null
}

export const buildNoteOgImageUrl = (title: string, summary: string) => {
  const baseUrl = 'https://og-image.vercel.app/'
  const encodedTitle = encodeURIComponent(title)
  const encodedDescription = encodeURIComponent(summary)
  return `${baseUrl}${encodedTitle}.png?theme=light&md=1&fontSize=75px&images=https%3A%2F%2Fraw.githubusercontent.com%2Fjiunbae%2Fjiunbae.github.io%2Fmain%2Fstatic%2Fprofile.png&description=${encodedDescription}`
}
