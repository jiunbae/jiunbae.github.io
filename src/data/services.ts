export interface ServiceItem {
  slug?: string
  title: string
  description: string
  icon: string
  iconType: 'image' | 'letter'
  tags: string[]
  status: 'live' | 'beta' | 'internal'
  category: 'ai' | 'dev-tools' | 'lifestyle' | 'infra'
  github?: string
  post?: string
}

const services: ServiceItem[] = [
  {
    title: 'Kurim',
    description: 'AI-powered branding for developers. Generate logos, landing pages, and brand identity from idea to deployed site.',
    icon: '/images/services/kurim.ico',
    iconType: 'image',
    tags: ['Next.js', 'FastAPI', 'OpenAI', 'PostgreSQL'],
    status: 'beta',
    category: 'ai',
  },
  {
    title: 'Kongbu',
    description: 'AI code learning platform. Converts GitHub repos into interactive quizzes to help engineers understand codebases faster.',
    icon: '/images/services/kongbu.svg',
    iconType: 'image',
    tags: ['Next.js', 'Express', 'Azure OpenAI', 'PostgreSQL'],
    status: 'beta',
    category: 'ai',
  },
  {
    slug: 'https://claude.jiun.dev',
    title: 'Claude Code Cloud',
    description: 'Web-based IDE for running Claude Code in the cloud. Real-time terminal, file explorer, and session management.',
    icon: '/images/services/claude-code-cloud.svg',
    iconType: 'image',
    tags: ['Next.js', 'WebSocket', 'xterm.js', 'Monaco'],
    status: 'beta',
    category: 'dev-tools',
    github: 'https://github.com/jiunbae/claude-code-cloud',
    post: '/claude-code-cloud',
  },
  {
    slug: 'https://issue.jiun.dev',
    title: 'IssueBoard',
    description: 'Lightweight issue tracking system with real-time collaboration. A minimal alternative to Linear and Jira.',
    icon: '/images/services/issueboard.svg',
    iconType: 'image',
    tags: ['Rust', 'Axum', 'React', 'PostgreSQL'],
    status: 'beta',
    category: 'dev-tools',
  },
  {
    slug: 'https://selectchatgpt.jiun.dev',
    title: 'SelectChatGPT',
    description: 'Chrome extension for ChatGPT. Select specific messages to create filtered share links or export as markdown.',
    icon: '/images/services/selectchatgpt.svg',
    iconType: 'image',
    tags: ['Plasmo', 'React', 'Express', 'MongoDB'],
    status: 'live',
    category: 'dev-tools',
    github: 'https://github.com/jiunbae/select-chat-gpt',
  },
  {
    slug: 'https://ssudam.jiun.dev',
    title: 'Ssudam',
    description: 'iOS app for tracking daily-use consumables. Monitor usage periods, replacement timing, and spending analytics.',
    icon: '/images/services/ssudam.png',
    iconType: 'image',
    tags: ['SwiftUI', 'FastAPI', 'PostgreSQL', 'CloudKit'],
    status: 'live',
    category: 'lifestyle',
  },
  {
    title: 'Mstoon',
    description: 'Self-hosted webtoon viewer for browsing and reading web comics with a clean reading experience.',
    icon: 'M',
    iconType: 'letter',
    tags: ['Nginx', 'SQLite', 'Docker'],
    status: 'live',
    category: 'lifestyle',
  },
  {
    title: 'Aily',
    description: 'Agent bridge dashboard with Discord integration. Manage notifications and automate workflows across services.',
    icon: 'A',
    iconType: 'letter',
    tags: ['Dashboard', 'Discord', 'SQLite'],
    status: 'beta',
    category: 'infra',
    github: 'https://github.com/jiunbae/aily',
  },
  {
    slug: 'https://tokka.jiun.dev',
    title: 'Tokka',
    description: 'Token analytics and monitoring service. Track and visualize token usage across AI-powered applications.',
    icon: 'T',
    iconType: 'letter',
    tags: ['Docker', 'Kubernetes'],
    status: 'beta',
    category: 'infra',
  },
]

export const categories = {
  ai: { label: 'AI', color: '#a78bfa' },
  'dev-tools': { label: 'Dev Tools', color: '#60a5fa' },
  lifestyle: { label: 'Lifestyle', color: '#34d399' },
  infra: { label: 'Infra', color: '#f97316' },
} as const

export default services
