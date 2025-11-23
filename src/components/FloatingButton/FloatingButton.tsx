import clsx from 'clsx'

import { ArrowUpIcon } from '@/components/icons'

import * as styles from './FloatingButton.module.scss'
import { useFloatingButton } from './hooks'

export const FloatingButton = () => {
  const { isVisible, scrollToTop } = useFloatingButton()

  return (
    <button
      className={clsx(styles.floatingButton, { [styles.visible]: isVisible })}
      onClick={scrollToTop}
      aria-label="Scroll to top"
      aria-hidden={!isVisible}
    >
      <ArrowUpIcon className={styles.arrowUpIcon} aria-hidden="true" />
    </button>
  )
}
