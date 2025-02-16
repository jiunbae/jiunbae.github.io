import clsx from 'clsx'

import ArrowUpIcon from '@/images/icons/arrow-up.svg'

import * as styles from './FloatingButton.module.scss'
import { useFloatingButton } from './hooks'

export const FloatingButton = () => {
  const { isVisible, scrollToTop } = useFloatingButton()

  return (
    <button className={clsx(styles.floatingButton, { [styles.visible]: isVisible })} onClick={scrollToTop}>
      <ArrowUpIcon className={styles.arrowUpIcon} />
    </button>
  )
}
