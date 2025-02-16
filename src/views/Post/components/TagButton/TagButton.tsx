import clsx from 'clsx'
import { navigate } from 'gatsby'
import * as styles from './TagButton.module.scss'

type TagButtonProps = {
  name: string;
};

export const TagButton = ({ name }: TagButtonProps) => {
  const handleClick = () => {
    navigate('/', { state: { tag: name } })
  }

  return (
    <div className={clsx(styles.tag)} onClick={handleClick}>
      <span>{name}</span>
    </div>
  )
}
