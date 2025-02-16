import clsx from 'clsx'
import { type ReactNode } from 'react'

import * as styles from './IconWrapper.module.scss'

interface IconWrapperProps {
  href: string;
  children: ReactNode;
  className?: string;
};

export const IconWrapper = ({ href, children, className }: IconWrapperProps) => (
  <a href={href} target="_blank" className={clsx(styles.profileIcon, className)} rel="noreferrer">
    {children}
  </a>
)
