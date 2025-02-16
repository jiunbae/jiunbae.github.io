import TableOfContentsIcon from '@/images/icons/table-of-contents.svg'

import { useTocStyleObserver } from './hooks'
import * as styles from './TableOfContents.module.scss'

type TableOfContentsProps = {
  html: string;
};

export const TableOfContents = ({ html }: TableOfContentsProps) => {
  const { ref } = useTocStyleObserver()

  return (
    <div className={styles.wrapper}>
      <TableOfContentsIcon className={styles.TableOfContentsIcon} />
      <div ref={ref} className={styles.tableOfContents} dangerouslySetInnerHTML={{ __html: html }}></div>
    </div>
  )
}
