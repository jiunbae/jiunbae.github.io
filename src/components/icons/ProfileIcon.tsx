import type { IconProps } from './RssIcon'

export const ProfileIcon = ({ size = 24, width, height, stroke = 'currentColor', fill = 'none', ...rest }: IconProps) => {
  return (
    <svg
      width={width ?? size}
      height={height ?? size}
      viewBox="0 0 24 24"
      fill={fill}
      xmlns="http://www.w3.org/2000/svg"
      {...rest}
    >
      <path d="M18 20a6 6 0 0 0-12 0" stroke={stroke} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={12} cy={10} r={4} stroke={stroke} strokeWidth={2} />
      <circle cx={12} cy={12} r={10} stroke={stroke} strokeWidth={2} />
    </svg>
  )
}
