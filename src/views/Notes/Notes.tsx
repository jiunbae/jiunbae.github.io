import type { HeadProps, PageProps } from 'gatsby'
import { useMemo } from 'react'

import { FloatingButton, Seo } from '@/components'
import { getRefinedStringValue } from '@/utils'

import { TagList } from '../Home/components'
import { TAGS } from '../Home/constants'
import { useTag } from '../Home/hooks/useTag'
import { NoteList } from './components'
import * as styles from './Notes.module.scss'

interface LocationState {
  tag?: string;
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

  return (
    <main className={styles.main}>
      <TagList tags={tags} selectedTag={selectedTag} clickTag={clickTag} className={styles.tagList} />
      <NoteList notes={visibleNotes} className={styles.noteList} />
      <FloatingButton />
    </main>
  )
}

export const Head = ({ location: { pathname }, data: { site, file } }: HeadProps<Queries.NotesQuery>) => {
  const seo = {
    title: `${getRefinedStringValue(site?.siteMetadata.title)} | Notes`,
    description: 'Short notes and quick updates from Jiunbae.',
    heroImage: getRefinedStringValue(file?.publicURL)
  }

  return <Seo title={seo.title} description={seo.description} heroImage={seo.heroImage} pathname={pathname} />
}

export default Notes
