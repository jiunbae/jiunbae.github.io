export interface ServiceItem {
  slug?: string
  title: string
  /** 최근 활동일(YYYY-MM-DD, 마지막 커밋 기준) — 최근 활동순 정렬 키 */
  updated: string
  description: string
  icon: string
  iconType: 'image' | 'letter'
  tags: string[]
  status: 'live' | 'beta' | 'internal'
  category: 'ai' | 'dev-tools' | 'lifestyle' | 'infra'
  github?: string
  post?: string
  /** npm 패키지명 (예: '@open330/agt') — npmjs 링크로 렌더링 */
  npm?: string
  /** App Store / Mac App Store 링크 */
  store?: string
}

// 정렬은 projects.astro에서 category(ai→dev-tools→lifestyle→infra) →
// status(live→beta→internal) → title 순으로 결정적 처리. 아래 배열 순서는 무관.
const services: ServiceItem[] = [
  // ── AI ──────────────────────────────────────────────
  {
    slug: 'https://tokka.jiun.dev',
    title: 'Tokka',
    updated: '2026-05-05',
    description: '카카오톡 대화 분석 서비스. 채팅 내보내기에서 AI 페르소나 프로파일링과 관계·대화 통계를 뽑아냅니다.',
    icon: '/images/services/tokka.svg',
    iconType: 'image',
    tags: ['React', 'FastAPI', 'Gemini AI', 'Kubernetes'],
    status: 'live',
    category: 'ai',
    post: '/posts/tokka-kakaotalk-analysis/',
  },
  {
    slug: 'https://jiun.dev/ai-horoscope/',
    title: 'AI 운세 (제피로스)',
    updated: '2026-07-06',
    description: '매일 아침 AI 모델들의 운세를 전하는 디지털 무당 제피로스.',
    icon: '/images/services/ai-horoscope.png',
    iconType: 'image',
    tags: ['Python', 'Gemini API', 'GitHub Pages', 'Gitea Actions'],
    status: 'live',
    category: 'ai',
    github: 'https://github.com/jiunbae/ai-horoscope',
  },
  {
    slug: 'https://burstpick.app',
    title: 'BurstPick',
    updated: '2026-03-29',
    description: '연사 사진을 AI로 골라내는 macOS 컬링 앱. Mac App Store 배포.',
    icon: '/images/services/burstpick.png',
    iconType: 'image',
    tags: ['Swift', 'macOS', 'AI'],
    status: 'live',
    category: 'ai',
    post: '/posts/burstpick-ai-agent-reviewers/',
  },
  {
    slug: 'https://flatten.jiun.dev',
    title: 'Flatten',
    updated: '2026-07-18',
    description: 'AI를 어디까지 설득할 수 있나. 근거·반론·대화 전략으로 겨루는 설득 게임.',
    icon: '/images/services/flatten.png',
    iconType: 'image',
    tags: ['AI', 'LLM', 'Game'],
    status: 'live',
    category: 'ai',
  },
  {
    slug: 'https://finchi.jiun.dev',
    title: 'Finchi',
    updated: '2026-07-18',
    description: '경제 퀴즈로 금융 IQ 올리기. 한국은행·금감원·시중은행 리포트 기반 학습 플랫폼.',
    icon: '/images/services/finchi.png',
    iconType: 'image',
    tags: ['AI', '퀴즈', '금융'],
    status: 'live',
    category: 'ai',
  },
  {
    slug: 'https://nolbul.jiun.dev',
    title: 'Nolbul',
    updated: '2026-04-05',
    description: '멀티에이전트 협동을 연구하는 카드게임 플랫폼. AI 에이전트들이 함께 플레이합니다.',
    icon: 'N',
    iconType: 'letter',
    tags: ['Multi-Agent', 'AI', '카드게임'],
    status: 'live',
    category: 'ai',
    github: 'https://github.com/jiunbae/nolbul',
  },
  {
    slug: 'https://kiwimu.jiun.dev',
    title: 'kiwimu',
    updated: '2026-07-11',
    description: '교재·PDF를 서로 링크된 학습 위키와 퀴즈로 변환.',
    icon: 'k',
    iconType: 'letter',
    tags: ['CLI', 'npm', 'LLM'],
    status: 'live',
    category: 'ai',
    npm: '@open330/kiwimu',
    github: 'https://github.com/Open330/kiwimu',
    post: '/posts/kiwimu-14-persona-review/',
  },
  {
    title: 'Kurim',
    updated: '2026-05-04',
    description: '개발자를 위한 AI 브랜딩. 아이디어에서 로고·랜딩·브랜드 아이덴티티까지 배포된 사이트로.',
    icon: '/images/services/kurim.ico',
    iconType: 'image',
    tags: ['Next.js', 'FastAPI', 'OpenAI', 'PostgreSQL'],
    status: 'beta',
    category: 'ai',
    slug: 'https://kurim.jiun.dev',
  },
  {
    title: 'Kongbu',
    updated: '2026-07-03',
    description: 'GitHub 저장소를 퀴즈로 바꿔, 배포하는 코드를 진짜로 이해하게 합니다.',
    icon: '/images/services/kongbu.svg',
    iconType: 'image',
    tags: ['Next.js', 'Express', 'Azure OpenAI', 'PostgreSQL'],
    status: 'beta',
    category: 'ai',
    slug: 'https://kongbu.jiun.dev',
  },
  {
    slug: 'https://arxiblog.jiun.dev',
    title: 'arxiblog',
    updated: '2026-06-27',
    description: 'arXiv 논문을 주석 달린 블로그 글로 변환.',
    icon: '/images/services/arxiblog.svg',
    iconType: 'image',
    tags: ['Python', 'LLM', 'arXiv'],
    status: 'beta',
    category: 'ai',
  },

  // ── Dev Tools ───────────────────────────────────────
  {
    slug: 'https://selectchatgpt.jiun.dev',
    title: 'SelectChatGPT',
    updated: '2026-03-15',
    description: 'ChatGPT용 크롬 확장. 메시지를 골라 공유 링크를 만들거나 마크다운으로 내보냅니다.',
    icon: '/images/services/selectchatgpt.svg',
    iconType: 'image',
    tags: ['Plasmo', 'React', 'Express', 'MongoDB'],
    status: 'live',
    category: 'dev-tools',
    github: 'https://github.com/jiunbae/select-chat-gpt',
  },
  {
    slug: 'https://prompt.jiun.dev',
    title: 'Oh My Prompt',
    updated: '2026-07-16',
    description: 'AI 코딩 세션의 프롬프트 저널. 기록하고 되짚는 CLI + 웹.',
    icon: '/images/services/oh-my-prompt.svg',
    iconType: 'image',
    tags: ['CLI', 'npm', 'AI'],
    status: 'live',
    category: 'dev-tools',
    npm: 'oh-my-prompt',
    github: 'https://github.com/jiunbae/oh-my-prompt',
    post: '/posts/oh-my-prompt-ai-prompt-journal/',
  },
  {
    title: 'agt',
    updated: '2026-05-22',
    description: 'AI 코딩 에이전트용 스킬·페르소나 패키지 매니저.',
    icon: '/images/services/agt.png',
    iconType: 'image',
    tags: ['CLI', 'npm', 'Agents'],
    status: 'live',
    category: 'dev-tools',
    npm: '@open330/agt',
    github: 'https://github.com/Open330/agt',
    post: '/posts/agt-package-manager-for-ai-agents/',
  },
  {
    slug: 'https://jiun.dev/OTPeek/',
    title: 'OTPeek',
    updated: '2026-07-07',
    description: '크로스플랫폼 2FA·OTP 앱. 위젯과 Rust 코어.',
    icon: '/images/services/otpeek.png',
    iconType: 'image',
    tags: ['Rust', 'Cross-platform', 'OTP'],
    status: 'live',
    category: 'dev-tools',
    github: 'https://github.com/jiunbae/otpeek',
    store: 'https://apps.apple.com/app/otpeek/id6787845951',
    post: '/posts/otpeek-cross-platform-otp/',
  },
  {
    title: 'BarShelf',
    updated: '2026-07-09',
    slug: 'https://barshelf.jiun.dev',
    description: 'macOS 메뉴바 위젯 허브. OTP·LLM 사용량·CI 상태를 한눈에. 노터라이즈 릴리스.',
    icon: '/images/services/barshelf.png',
    iconType: 'image',
    tags: ['Swift', 'macOS', 'Menu Bar'],
    status: 'live',
    category: 'dev-tools',
    github: 'https://github.com/Open330/barshelf',
    post: '/posts/barshelf-macos-menubar-hub/',
  },
  {
    title: 'IssueBoard',
    updated: '2026-03-15',
    description: '실시간 협업 이슈 트래커. Linear·Jira의 미니멀 대안.',
    icon: '/images/services/issueboard.svg',
    iconType: 'image',
    tags: ['Rust', 'Axum', 'React', 'PostgreSQL'],
    status: 'beta',
    category: 'dev-tools',
    slug: 'https://issue.jiun.dev',
  },
  {
    title: 'Claude Code Cloud',
    updated: '2026-03-15',
    description: '브라우저에서 Claude Code를 돌리는 웹 IDE. 실시간 터미널·파일 탐색기·세션 관리.',
    icon: '/images/services/claude-code-cloud.svg',
    iconType: 'image',
    tags: ['Next.js', 'WebSocket', 'xterm.js', 'Monaco'],
    status: 'internal',
    category: 'dev-tools',
    github: 'https://github.com/jiunbae/claude-code-cloud',
    post: '/posts/claude-code-cloud/',
  },

  // ── Lifestyle ───────────────────────────────────────
  {
    slug: 'https://ssudam.jiun.dev',
    title: 'Ssudam',
    updated: '2026-03-20',
    description: '생활 소모품 관리 iOS 앱. 사용 주기·교체 시점·지출을 추적.',
    icon: '/images/services/ssudam.png',
    iconType: 'image',
    tags: ['SwiftUI', 'FastAPI', 'PostgreSQL', 'CloudKit'],
    status: 'live',
    category: 'lifestyle',
  },
  {
    title: 'Mstoon',
    updated: '2026-01-18',
    description: '직접 호스팅하는 웹툰 뷰어. 군더더기 없는 읽기 경험.',
    icon: 'M',
    iconType: 'letter',
    tags: ['Nginx', 'SQLite', 'Docker'],
    status: 'live',
    category: 'lifestyle',
  },
  {
    slug: 'https://nova-pouch.jiun.dev',
    title: 'Nova Pouch',
    updated: '2026-03-16',
    description: '매일 단어 조각을 뽑아 상상의 세계를 쓰는 데일리 퍼즐.',
    icon: '/images/services/nova-pouch.png',
    iconType: 'image',
    tags: ['TypeScript', 'Puzzle', 'Daily'],
    status: 'live',
    category: 'lifestyle',
    post: '/posts/nova-pouch-daily-word-puzzle/',
  },
  {
    slug: 'https://chartlog.jiun.dev',
    title: 'Chartlog',
    updated: '2026-06-27',
    description: '친구방이 곧 하나의 차트. 사이버머니로 같이 매매하는 소셜 트레이딩 게임.',
    icon: '/images/services/chartlog.png',
    iconType: 'image',
    tags: ['Trading', 'Social', 'Game'],
    status: 'live',
    category: 'lifestyle',
    post: '/posts/chartlog-social-trading-game/',
  },
  {
    title: 'Stashbar',
    updated: '2026-07-08',
    slug: 'https://jiun.dev/stashbar/',
    description: '최근 파일을 쌓아두는 macOS 메뉴바 스택 앱.',
    icon: 'S',
    iconType: 'letter',
    tags: ['Swift', 'macOS', 'Menu Bar'],
    status: 'live',
    category: 'lifestyle',
    github: 'https://github.com/jiunbae/file-stack',
  },
  {
    slug: 'https://domidman.jiun.dev',
    title: '도믿맨: 퇴근길 생존기',
    updated: '2026-07-18',
    description: '퇴근길 생존 풍자 게임. 길에서 만난 NPC를 피하거나 도우며 집까지 걸어갑니다.',
    icon: '/images/services/domidman.png',
    iconType: 'image',
    tags: ['Vanilla JS', 'Canvas', 'Hono API', 'MongoDB'],
    status: 'beta',
    category: 'lifestyle',
  },
  {
    slug: 'https://bubbles.jiun.dev',
    title: 'Bubbles',
    updated: '2026-07-18',
    description: '비흡연자를 위한 담타. 3D 멀티플레이어 버블 게임.',
    icon: 'B',
    iconType: 'letter',
    tags: ['Three.js', '3D', '멀티플레이'],
    status: 'beta',
    category: 'lifestyle',
    post: '/posts/bubbles-multiplayer-bubble-game/',
  },

  // ── Infra ───────────────────────────────────────────
  {
    title: 'Aily',
    updated: '2026-07-12',
    description: 'Discord로 연결하는 에이전트 브리지 대시보드. 알림과 워크플로를 한곳에서.',
    icon: '/images/services/aily.svg',
    iconType: 'image',
    tags: ['Dashboard', 'Discord', 'SQLite'],
    status: 'beta',
    category: 'infra',
    github: 'https://github.com/jiunbae/aily',
    post: '/posts/aily-ai-session-bridge/',
  },
]

// 카테고리 컬러는 다크/라이트 모두에서 4.5:1+ contrast 만족하도록 darker shade 선택.
export const categories = {
  ai: { label: 'AI', color: '#7c3aed' },          /* violet-600 — 5.6:1 on white */
  'dev-tools': { label: 'Dev Tools', color: '#2563eb' }, /* blue-600 — 5.17:1 */
  lifestyle: { label: 'Lifestyle', color: '#047857' },   /* green-700 — 5.5:1 */
  infra: { label: 'Infra', color: '#c2410c' },     /* orange-700 — 4.9:1 */
} as const

// 결정적 정렬용 순서 맵
export const categoryOrder: Record<ServiceItem['category'], number> = {
  ai: 0,
  'dev-tools': 1,
  lifestyle: 2,
  infra: 3,
}

export const statusOrder: Record<ServiceItem['status'], number> = {
  live: 0,
  beta: 1,
  internal: 2,
}

export default services
