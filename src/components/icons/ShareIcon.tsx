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
      <circle cx="18" cy="5" r="3" fill={fill} />
      <circle cx="6" cy="12" r="3" fill={fill} />
      <circle cx="18" cy="19" r="3" fill={fill} />
      <path
        d="M8.59 13.51L15.42 17.49M15.41 6.51L8.59 10.49"
        stroke={stroke}
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  )
}
