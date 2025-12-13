import type { IconProps } from './RssIcon'

export const ShareIcon = ({ size = 24, width, height, fill = 'currentColor', stroke = 'currentColor', ...rest }: IconProps) => {
  return (
    <svg
      width={width ?? size}
      height={height ?? size}
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      {...rest}
    >
      <circle cx="18" cy="5" r="2.5" fill={fill} />
      <circle cx="6" cy="12" r="2.5" fill={fill} />
      <circle cx="18" cy="19" r="2.5" fill={fill} />
      <path
        d="M8.5 13.5L15.5 17.5M15.5 6.5L8.5 10.5"
        stroke={stroke}
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  )
}
