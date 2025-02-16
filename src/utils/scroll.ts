export const optimizedScroll = <T extends (...args: any[]) => void>(callback: T): T => {
  let ticking = false

  const throttled = ((...args: Parameters<T>) => {
    if (!ticking) {
      window.requestAnimationFrame(() => {
        callback(...args)
        ticking = false
      })
      ticking = true
    }
  }) as T

  return throttled
}
