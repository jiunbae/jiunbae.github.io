/**
 * 성능 최적화 유틸리티
 * Throttle, Debounce 등의 성능 최적화 함수
 */

/**
 * Throttle 함수
 * 지정된 시간 동안 함수가 한 번만 실행되도록 제한
 * 스크롤, 리사이즈 이벤트 등에 유용
 *
 * @param func 실행할 함수
 * @param limit 제한 시간 (밀리초)
 * @returns throttled 함수
 *
 * @example
 * const handleScroll = throttle(() => {
 *   console.log('Scrolled!')
 * }, 100)
 *
 * window.addEventListener('scroll', handleScroll)
 */
export const throttle = <T extends (...args: any[]) => any>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean = false
  let lastResult: ReturnType<T>

  return function (this: any, ...args: Parameters<T>): void {
    if (!inThrottle) {
      lastResult = func.apply(this, args)
      inThrottle = true
      setTimeout(() => {
        inThrottle = false
      }, limit)
    }
  }
}

/**
 * Debounce 함수
 * 함수 호출을 지연시키고, 마지막 호출 이후 일정 시간이 지나야 실행
 * 검색 입력, 윈도우 리사이즈 등에 유용
 *
 * @param func 실행할 함수
 * @param wait 대기 시간 (밀리초)
 * @returns debounced 함수
 *
 * @example
 * const handleSearch = debounce((query: string) => {
 *   console.log('Searching for:', query)
 * }, 300)
 *
 * input.addEventListener('input', (e) => handleSearch(e.target.value))
 */
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: ReturnType<typeof setTimeout> | null = null

  return function (this: any, ...args: Parameters<T>): void {
    const later = () => {
      timeout = null
      func.apply(this, args)
    }

    if (timeout !== null) {
      clearTimeout(timeout)
    }
    timeout = setTimeout(later, wait)
  }
}

/**
 * RequestAnimationFrame을 사용한 Throttle
 * 브라우저의 repaint 주기에 맞춰 함수 실행 제한
 * 애니메이션이나 스크롤 핸들러에 최적화
 *
 * @param func 실행할 함수
 * @returns throttled 함수
 *
 * @example
 * const handleScroll = rafThrottle(() => {
 *   console.log('Scrolled!')
 * })
 *
 * window.addEventListener('scroll', handleScroll)
 */
export const rafThrottle = <T extends (...args: any[]) => any>(
  func: T
): ((...args: Parameters<T>) => void) => {
  let rafId: number | null = null

  return function (this: any, ...args: Parameters<T>): void {
    if (rafId !== null) {
      return
    }

    rafId = requestAnimationFrame(() => {
      func.apply(this, args)
      rafId = null
    })
  }
}

/**
 * 특정 시간 동안 대기하는 Promise
 * async/await와 함께 사용
 *
 * @param ms 대기 시간 (밀리초)
 * @returns Promise
 *
 * @example
 * async function fetchData() {
 *   await delay(1000)
 *   console.log('1초 후 실행')
 * }
 */
export const delay = (ms: number): Promise<void> => {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/**
 * 함수 실행 시간 측정
 * 개발 중 성능 디버깅용
 *
 * @param func 측정할 함수
 * @param label 라벨 (콘솔에 표시)
 * @returns 함수 실행 결과
 *
 * @example
 * const result = await measurePerformance(
 *   () => expensiveCalculation(),
 *   'Expensive Calculation'
 * )
 */
export const measurePerformance = async <T>(
  func: () => T | Promise<T>,
  label: string
): Promise<T> => {
  const start = performance.now()
  const result = await func()
  const end = performance.now()

  if (process.env.NODE_ENV === 'development') {
    console.log(`[Performance] ${label}: ${(end - start).toFixed(2)}ms`)
  }

  return result
}
