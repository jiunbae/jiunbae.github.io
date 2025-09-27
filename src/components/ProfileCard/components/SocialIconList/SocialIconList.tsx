import type { ComponentType } from 'react'

import {
  MailIcon,
  FacebookIcon,
  LinkedinIcon,
  GithubIcon,
  TwitterIcon,
  InstagramIcon
} from '@/components/icons'
import type { IconProps } from '@/components/icons'

import { IconWrapper } from '../IconWrapper'

export const SOCIAL_TYPES = ['email', 'facebook', 'linkedin', 'github', 'twitter', 'instagram'] as const
export type SocialType = typeof SOCIAL_TYPES[number]

type IconType = ComponentType<IconProps>

const SOCIAL_ICONS: Record<SocialType, IconType> = {
  email: MailIcon,
  facebook: FacebookIcon,
  linkedin: LinkedinIcon,
  github: GithubIcon,
  twitter: TwitterIcon,
  instagram: InstagramIcon
} as const

interface SocialIconItem {
  IconComponent: IconType;
  href: string;
}

interface SocialIconListProps {
  social: Record<SocialType, string>
}

const SocialIcon = ({ IconComponent, href }: SocialIconItem) => (
  <IconWrapper href={href}>
    <IconComponent size={20} />
  </IconWrapper>
)

export const SocialIconList = ({ social }: SocialIconListProps) => (
  <>
    {Object.entries(social).map(([key, value]) => (
      <SocialIcon key={key} IconComponent={SOCIAL_ICONS[key as SocialType]} href={value} />
    ))}
  </>
)
