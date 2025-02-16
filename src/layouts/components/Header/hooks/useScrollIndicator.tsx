import { useCallback, useEffect, useState } from 'react'

import { optimizedScroll } from '@/utils'

export const useScrollIndicator = (pathname: string) => {
  const isPost = pathname.includes('/posts/')
  const [progressWidth, setProgressWidth] = useState<number>(0)

  const updateProgress = useCallback(() => {
    const { scrollY, innerHeight } = window
    const { scrollHeight } = document.documentElement
    const maxScroll = scrollHeight - innerHeight
    const progress = maxScroll > 0 ? Math.min((scrollY / maxScroll) * 100, 100) : 100
    setProgressWidth(progress)
  }, [])

  useEffect(() => {
    if (!isPost) return

    const handleScroll = optimizedScroll(updateProgress)
    window.addEventListener('scroll', handleScroll)
    window.addEventListener('resize', handleScroll)
    
    updateProgress()

    return () => {
      window.removeEventListener('scroll', handleScroll)
      window.removeEventListener('resize', handleScroll)
    }
  }, [isPost, updateProgress])

  return { isPost, progressWidth }
}
