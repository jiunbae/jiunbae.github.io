import clsx from 'clsx'

import * as styles from './Description.module.scss'

interface DescriptionProps {
  className?: string;
};

export const Description = ({ className }: DescriptionProps) => (
  <p className={clsx(styles.description, className)}>
    Driven machine learning enthusiast specializing in natural language processing and computer vision, committed to
    solving real-world problems through data-driven innovation.
  </p>
)
