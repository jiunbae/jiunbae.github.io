import * as styles from './Heading.module.scss'

interface HeadingProps {
  text: string;
  subText?: string;
};

export const Heading = ({ text, subText }: HeadingProps) => {
  return (
    <h2 className={styles.heading}>
      {text}
      {subText && (
        <span className={styles.subText}>
          {subText}
        </span>
      )}
    </h2>
  )
}
