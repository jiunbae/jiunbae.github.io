import type { IconProps } from './RssIcon'

export const TableOfContentsIcon = ({ size = 24, width, height, stroke = 'currentColor', fill = 'none', ...rest }: IconProps) => {
  return (
    <svg
      width={width ?? size}
      height={height ?? size}
      viewBox="0 0 24 24"
      fill={fill}
      xmlns="http://www.w3.org/2000/svg"
      {...rest}
    >
      <rect x={4} y={4} width={16} height={16} rx={2} stroke={stroke} strokeWidth={2} />
      <path d="M8 8h8" stroke={stroke} strokeWidth={2} strokeLinecap="round" />
      <path d="M8 12h8" stroke={stroke} strokeWidth={2} strokeLinecap="round" />
      <path d="M8 16h5" stroke={stroke} strokeWidth={2} strokeLinecap="round" />
    </svg>
  )
}
