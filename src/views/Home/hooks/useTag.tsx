import type { KeyboardEvent, MouseEvent } from 'react'
import { useCallback, useMemo, useState } from 'react'
import { navigate } from 'gatsby'

import { TAGS } from '../constants'

export const useTag = (
  totalCount: number,
  group: Queries.HomeQuery['allMarkdownRemark']['group'],
  defaultSelectedTag?: string
) => {
  const tags = useMemo(
    () => [{ fieldValue: TAGS.ALL, totalCount }, ...group].sort((a, b) => b.totalCount - a.totalCount),
    [group, totalCount]
  )
  const [selectedTag, setSelectedTag] = useState<string>(defaultSelectedTag ?? TAGS.ALL)
  const clickTag = useCallback(({ target }: MouseEvent<HTMLElement> | KeyboardEvent<HTMLElement>) => {
    if (!(target instanceof HTMLElement)) return
    const tagItem = target.closest('li')
    const tag = tagItem?.dataset.tag
    if (tag) {
      setSelectedTag(tag)
      navigate('/', { replace: true, state: { tag } })
    }
  }, [])

  return { tags, selectedTag, clickTag }
}
