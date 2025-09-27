import type { HeadProps, PageProps } from 'gatsby'
import { useEffect, useMemo } from 'react'

import { FloatingButton, Seo } from '@/components'
import { getRefinedStringValue } from '@/utils'

import { TagList } from '../Home/components'
import { TAGS } from '../Home/constants'
import { useTag } from '../Home/hooks/useTag'
import { NoteList } from './components'
import * as styles from './Notes.module.scss'
import { extractSlugFromLocation, sanitizeNoteSlug } from './utils'

interface LocationState {
  tag?: string;
  note?: string;
}

const Notes = ({ data, location }: PageProps<Queries.NotesQuery, object, LocationState>) => {
  const { nodes: allNotes, totalCount, group } = data.allMarkdownRemark
  const { tags, selectedTag, clickTag } = useTag(totalCount, group, location.state?.tag, { pathname: '/notes/' })

  const visibleNotes = useMemo(
    () => allNotes.filter(({ frontmatter }) => {
      if (selectedTag === TAGS.ALL) return true

      const noteTags = frontmatter.tags ?? []
      return noteTags.some(tag => tag === selectedTag)
    }),
    [allNotes, selectedTag]
  )

  useEffect(() => {
    if (typeof window === 'undefined') return

    const slugFromState = location.state?.note ? sanitizeNoteSlug(location.state.note) : null
    const slugFromLocation = extractSlugFromLocation(location.search, location.hash)
    const targetSlug = slugFromState || slugFromLocation

    if (!targetSlug) return

    const target = document.querySelector<HTMLElement>(`[data-note-slug='${targetSlug}']`)

    if (!target) return

    const highlight = () => {
      target.setAttribute('data-highlighted', 'true')
      target.scrollIntoView({ behavior: 'smooth', block: 'start' })

      window.setTimeout(() => {
        target.removeAttribute('data-highlighted')
      }, 2500)
    }

    // Delay to ensure rendering complete
    const timer = window.setTimeout(highlight, 150)

    return () => {
      window.clearTimeout(timer)
    }
  }, [location.search, location.hash, location.state, visibleNotes])

  return (
    <main className={styles.main}>
      <TagList tags={tags} selectedTag={selectedTag} clickTag={clickTag} className={styles.tagList} />
      <NoteList notes={visibleNotes} className={styles.noteList} />
      <FloatingButton />
    </main>
  )
}

export const Head = ({ location, data: { site, file } }: HeadProps<Queries.NotesQuery>) => {
  const { href } = location as typeof location & { href?: string }
  const pageUrl = href ?? location.pathname
  const seo = {
    title: `${getRefinedStringValue(site?.siteMetadata.title)} | Notes`,
    description: 'Short notes and quick updates from Jiunbae.',
    heroImage: getRefinedStringValue(file?.publicURL)
  }

  return <Seo title={seo.title} description={seo.description} heroImage={seo.heroImage} pathname={pageUrl} />
}

export default Notes
