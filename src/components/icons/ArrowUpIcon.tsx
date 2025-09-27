import type { IconProps } from './RssIcon'

export const ArrowUpIcon = ({ size = 54, width, height, fill = 'white', ...rest }: IconProps) => {
  return (
    <svg
      width={width ?? size}
      height={height ?? size}
      viewBox="0 0 54 54"
      xmlns="http://www.w3.org/2000/svg"
      {...rest}
    >
      <path d="M26.9999 24.363L15.8624 35.5005L12.6809 32.319L26.9999 18L41.3189 32.319L38.1374 35.5005L26.9999 24.363Z" fill={fill} />
    </svg>
  )
}
