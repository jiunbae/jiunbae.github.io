import type { SVGProps } from 'react'

export type IconProps = SVGProps<SVGSVGElement> & {
  size?: number;
}

export const RssIcon = ({ size = 24, width, height, stroke = 'currentColor', fill = 'none', ...rest }: IconProps) => {
  return (
    <svg
      width={width ?? size}
      height={height ?? size}
      viewBox="0 0 24 24"
      fill={fill}
      xmlns="http://www.w3.org/2000/svg"
      {...rest}
    >
      <path
        d="M4 11a9 9 0 0 1 9 9"
        stroke={stroke}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M4 4a16 16 0 0 1 16 16"
        stroke={stroke}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx={5} cy={19} r={1} fill={stroke} />
    </svg>
  )
}
