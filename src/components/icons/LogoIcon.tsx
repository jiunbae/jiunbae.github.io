import type { IconProps } from './RssIcon'

export const LogoIcon = ({ size = 24, width, height, stroke = 'currentColor', fill = 'none', ...rest }: IconProps) => {
  return (
    <svg
      width={width ?? size}
      height={height ?? size}
      viewBox="0 0 24 24"
      fill={fill}
      xmlns="http://www.w3.org/2000/svg"
      {...rest}
    >
      <circle cx={12} cy={12} r={10} stroke={stroke} strokeWidth={2} fill="none" />
      <text
        x="50%"
        y="50%"
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="14"
        fontWeight="800"
        fill={stroke}
      >
        J
      </text>
    </svg>
  )
}
