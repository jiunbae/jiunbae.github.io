import type { KeyboardEvent, MouseEvent } from 'react'
import { useCallback, useMemo, useState } from 'react'
import { navigate } from 'gatsby'

import { TAGS } from '../constants'

type TagGroup = {
  fieldValue: string | null;
  totalCount: number;
};

type UseTagOptions = {
  pathname?: string;
};

export const useTag = (
  totalCount: number,
  group: ReadonlyArray<TagGroup>,
  defaultSelectedTag?: string,
  { pathname = '/' }: UseTagOptions = {}
) => {
  const tags = useMemo(() => {
    const normalizedGroup = Array.from(group, ({ fieldValue, totalCount: count }) => ({ fieldValue, totalCount: count }))
    return [{ fieldValue: TAGS.ALL, totalCount }, ...normalizedGroup].sort((a, b) => b.totalCount - a.totalCount)
  }, [group, totalCount])
  const [selectedTag, setSelectedTag] = useState<string>(defaultSelectedTag ?? TAGS.ALL)
  const clickTag = useCallback(({ target }: MouseEvent<HTMLElement> | KeyboardEvent<HTMLElement>) => {
    if (!(target instanceof HTMLElement)) return
    const tagItem = target.closest('li')
    const tag = tagItem?.dataset.tag
    if (tag) {
      setSelectedTag(tag)
      navigate(pathname, { replace: true, state: { tag } })
    }
  }, [pathname])

  return { tags, selectedTag, clickTag }
}
