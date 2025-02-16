import type { ComponentType, SVGProps } from 'react'

import MailIcon from '@/images/icons/mail.svg'
import FacebookIcon from '@/images/icons/facebook.svg'
import LinkedinIcon from '@/images/icons/linkedin.svg'
import GithubIcon from '@/images/icons/github.svg'
import TwitterIcon from '@/images/icons/twitter.svg'
import InstagramIcon from '@/images/icons/instagram.svg'

import { IconWrapper } from '../IconWrapper'

export const SOCIAL_TYPES = ['email', 'facebook', 'linkedin', 'github', 'twitter', 'instagram'] as const
export type SocialType = typeof SOCIAL_TYPES[number]

type IconType = ComponentType<SVGProps<SVGSVGElement>>

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
    <IconComponent />
  </IconWrapper>
)

export const SocialIconList = ({ social }: SocialIconListProps) => (
  <>
    {Object.entries(social).map(([key, value]) => (
      <SocialIcon key={key} IconComponent={SOCIAL_ICONS[key as SocialType]} href={value} />
    ))}
  </>
)
